using System;
using Meta.XR;
using System.IO;
using System.Text;
using UnityEngine;
using System.Linq;
using Unity.Sentis;
using MIConvexHull;
using System.Collections;
using System.Collections.Generic;
using Random = UnityEngine.Random;

[RequireComponent(typeof(ParticleSystem))]
public class PointCloudController : MonoBehaviour
{
    // Raycast origin and raycast manager
    [Header("Raycast Setup")]
    public Transform rayOriginAnchor;
    public EnvironmentRaycastManager raycastManager;

    // ONNX Model 
    [Header("Inference Settings")]
    public ModelAsset modelAsset;

    // Number of points to be captured with quantity, size and speed
    [Header("Scanning Settings")]
    private int raysPerScan = 100;
    private float scanRadius = 0.2f;
    private float scanInterval = 0.0f;
    private float deletionRadius = 0.1f;

    // Settings for Point Cloud Visualization
    [Header("Visual Settings")]
    public ParticleSystem scanParticles;
    public GameObject scanIndicator;
    public Color chairScanColor = Color.green;
    public Color roomScanColor = Color.white;

    // Lists to store each of the point clouds and labels
    private List<Vector3> _chairPoints = new List<Vector3>();
    private List<Vector3> _roomPoints = new List<Vector3>();
    private List<Vector3> _combinedPoints = new List<Vector3>();
    private List<int> _labels = new List<int>();

    // State machine to control scanning process
    private Coroutine _scanCoroutine;
    private enum ScanState { Idle, ScanningChair, ScanningRoom, ReadyToProcess }
    private ScanState _currentState = ScanState.Idle;

    // Mesh to visualize the point cloud
    private GameObject _generatedMeshObject;
    public Material generatedMeshMaterial;

    private void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("Model Asset is not assigned in the Inspector!");
            return;
        }
        PointCloudInference.Initialize(modelAsset);
    }

    private void OnDestroy()
    {
        PointCloudInference.Shutdown();
    }

    private void Update()
    {
        MoveScanIndicator();
        HandleInput();
        HandlePointDeletion();
    }

    // Move the scan indicator to where the user is looking
    private void MoveScanIndicator()
    {
        if (scanIndicator == null) return;

        bool isDeleting = OVRInput.Get(OVRInput.RawButton.LIndexTrigger);
        bool shouldShowIndicator = _currentState == ScanState.Idle || _currentState == ScanState.ScanningChair || _currentState == ScanState.ScanningRoom || isDeleting;

        if (!shouldShowIndicator)
        {
            scanIndicator.SetActive(false);
            return;
        }

        var centralRay = new Ray(rayOriginAnchor.position, rayOriginAnchor.forward);
        if (raycastManager.Raycast(centralRay, out var centerHit))
        {
            scanIndicator.SetActive(true);
            scanIndicator.transform.position = centerHit.point + centerHit.normal * 0.01f;
            scanIndicator.transform.rotation = Quaternion.LookRotation(-centerHit.normal);

            float distance = Vector3.Distance(rayOriginAnchor.position, centerHit.point);
            float currentRadius = isDeleting ? deletionRadius : distance * scanRadius;
            scanIndicator.transform.localScale = new Vector3(currentRadius * 2, currentRadius * 2, 1f);
        }
        else
        {
            scanIndicator.SetActive(false);
        }
    }

    // Input handling and state machine for user to control app
    private void HandleInput()
    {
        switch (_currentState)
        {   
            // Idle state (User is not doing anything)
            case ScanState.Idle:
                if (OVRInput.GetDown(OVRInput.RawButton.RIndexTrigger))
                {
                    _currentState = ScanState.ScanningChair;
                    _scanCoroutine = StartCoroutine(ScanningRoutine());
                }
                break;

            // User is scanning the chair
            case ScanState.ScanningChair:
                if (OVRInput.GetDown(OVRInput.RawButton.RIndexTrigger))
                {
                    if (_scanCoroutine == null) _scanCoroutine = StartCoroutine(ScanningRoutine());
                }

                if (OVRInput.GetUp(OVRInput.RawButton.RIndexTrigger))
                {
                    if (_scanCoroutine != null) StopCoroutine(_scanCoroutine); _scanCoroutine = null;
                }

                if (OVRInput.GetDown(OVRInput.RawButton.A))
                {
                    if (_scanCoroutine != null) StopCoroutine(_scanCoroutine);
                    _scanCoroutine = null;
                    _currentState = ScanState.ScanningRoom;
                    Debug.Log($"Chair scan complete with {_chairPoints.Count} points. Press and hold Trigger to scan the ROOM.");
                }
                break;

            // User is scanning the room/scene
            case ScanState.ScanningRoom:
                scanRadius = 0.5f;
                if (OVRInput.GetDown(OVRInput.RawButton.RIndexTrigger))
                {
                    if (_scanCoroutine == null) _scanCoroutine = StartCoroutine(ScanningRoutine());
                }
                if (OVRInput.GetUp(OVRInput.RawButton.RIndexTrigger))
                {
                    if (_scanCoroutine != null) StopCoroutine(_scanCoroutine); _scanCoroutine = null;
                }
                if (OVRInput.GetDown(OVRInput.RawButton.A))
                {
                    if (_scanCoroutine != null) StopCoroutine(_scanCoroutine);
                    _scanCoroutine = null;
                    _currentState = ScanState.ReadyToProcess;
                    Debug.Log($"Room scan complete with {_roomPoints.Count} points. Total points: {_chairPoints.Count + _roomPoints.Count}");
                    Debug.Log("Controls: 'A' = Save || 'B' = AI Segmentation");
                }
                break;

            // Allows user to choose 'Save' or 'Run'
            case ScanState.ReadyToProcess:
                if (OVRInput.GetDown(OVRInput.RawButton.A))
                {
                    SavePointCloud();
                }
                if (OVRInput.GetDown(OVRInput.RawButton.B))
                {
                    RunAISegmentation();
                }
                break;
        }

        // Clear the point cloud and reset app state
        if (OVRInput.GetDown(OVRInput.RawButton.RThumbstick))
        {
            scanRadius = 0.2f;
            ClearScan();
        }
    }


    // Allows user to delete points from the point cloud using scan indicator
    private void HandlePointDeletion()
    {
        if (OVRInput.Get(OVRInput.RawButton.LIndexTrigger))
        {
            var deletionRay = new Ray(rayOriginAnchor.position, rayOriginAnchor.forward);
            if (raycastManager.Raycast(deletionRay, out var hitInfo))
            {
                Vector3 hitPoint = hitInfo.point;
                int chairPointsRemoved = _chairPoints.RemoveAll(point => Vector3.Distance(point, hitPoint) < deletionRadius);
                int roomPointsRemoved = _roomPoints.RemoveAll(point => Vector3.Distance(point, hitPoint) < deletionRadius);
                if (chairPointsRemoved > 0 || roomPointsRemoved > 0)
                {
                    Debug.Log($"Removed {chairPointsRemoved} chair points and {roomPointsRemoved} room points.");
                    RefreshParticleSystem();
                }
            }
        }
    }

    // Refresh the particle system to match the current point cloud
    private void RefreshParticleSystem()
    {
        scanParticles.Clear();
        var particles = new List<ParticleSystem.Particle>();
        foreach (var point in _chairPoints)
        {
            particles.Add(new ParticleSystem.Particle
            {
                position = point,
                startSize = 0.01f,
                startLifetime = float.PositiveInfinity,
                startColor = chairScanColor
            });
        }
        foreach (var point in _roomPoints)
        {
            particles.Add(new ParticleSystem.Particle
            {
                position = point,
                startSize = 0.01f,
                startLifetime = float.PositiveInfinity,
                startColor = roomScanColor
            });
        }
        scanParticles.SetParticles(particles.ToArray(), particles.Count);
    }

    // Scanning routine for capturing point cloud data
    private IEnumerator ScanningRoutine()
    {
        bool isScanningChair = _currentState == ScanState.ScanningChair;
        Color scanColor = isScanningChair ? chairScanColor : roomScanColor;
        while (true)
        {
            PerformScan(scanColor, isScanningChair ? _chairPoints : _roomPoints);
            yield return new WaitForSeconds(scanInterval);
        }
    }

    // Control the amount of points sampled within scan indicator
    private void PerformScan(Color particleColor, List<Vector3> targetList)
    {
        var centralRay = new Ray(rayOriginAnchor.position, rayOriginAnchor.forward);
        if (!raycastManager.Raycast(centralRay, out var centerHit)) return;
        float distance = Vector3.Distance(rayOriginAnchor.position, centerHit.point);
        float currentRadius = distance * scanRadius;
        float goldenAngle = Mathf.PI * (3f - Mathf.Sqrt(5f));
        for (int i = 0; i < raysPerScan; i++)
        {
            float theta = i * goldenAngle;
            float r = Mathf.Sqrt(i) / Mathf.Sqrt(raysPerScan) * currentRadius;
            var pointOnCircle = new Vector2(Mathf.Cos(theta), Mathf.Sin(theta)) * r;
            Vector3 targetPoint = centerHit.point + (rayOriginAnchor.right * pointOnCircle.x) + (rayOriginAnchor.up * pointOnCircle.y);
            Ray scanRay = new Ray(rayOriginAnchor.position, (targetPoint - rayOriginAnchor.position).normalized);
            if (raycastManager.Raycast(scanRay, out var scanHit, distance + 0.5f))
            {
                targetList.Add(scanHit.point);
                EmitParticleAt(scanHit.point, particleColor);
            }
        }
    }

    // Place a particle at the scanned position
    private void EmitParticleAt(Vector3 position, Color color)
    {
        if (!scanParticles) return;
        var emitParams = new ParticleSystem.EmitParams
        {
            position = position,
            startSize = 0.01f,
            startLifetime = float.PositiveInfinity,
            startColor = color
        };
        scanParticles.Emit(emitParams, 1);
    }

    // Delete all particles and reset point cloud lists
    public void ClearScan()
    {
        if (_scanCoroutine != null) { StopCoroutine(_scanCoroutine); _scanCoroutine = null; }
        _chairPoints.Clear();
        _roomPoints.Clear();
        _combinedPoints.Clear();
        _labels.Clear();
        if (scanParticles) { scanParticles.Clear(); }
        // Destroy the generated mesh if it exists
        if (_generatedMeshObject != null)
        {
            Destroy(_generatedMeshObject);
        }
        _currentState = ScanState.Idle;
        Debug.Log("All data cleared. Press and hold Right Trigger to start scanning the CHAIR.");

    }

    // Save the point cloud to the local filesystem
    private void SavePointCloud()
    {
        if (_currentState != ScanState.ReadyToProcess) return;
        if (AssembleFinalPointCloud())
        {
            string fileName = $"PointCloud_{DateTime.Now:yyyyMMdd_HHmmss}.ply";
            PointCloudSaver.Save(_combinedPoints, _labels, fileName);
        }
    }

    // Assemble the final point cloud from chair and room
    private bool AssembleFinalPointCloud()
    {
        if (_chairPoints.Count == 0 && _roomPoints.Count == 0)
        {
            Debug.LogWarning("No points have been scanned. Nothing to process.");
            return false;
        }
        _combinedPoints.Clear();
        _labels.Clear();
        foreach (var point in _chairPoints) { _combinedPoints.Add(point); _labels.Add(1); }
        foreach (var point in _roomPoints) { _combinedPoints.Add(point); _labels.Add(0); }
        Debug.Log($"Assembled final point cloud: {_combinedPoints.Count} total points. ({_chairPoints.Count} chair, {_roomPoints.Count} room).");
        return true;
    }

    // Run the ONNX Model on the points
    private void RunAISegmentation()
    {
        AssembleFinalPointCloud();
        if (_combinedPoints.Count == 0)
        {
            Debug.LogWarning("No points to run inference on. Scan first.");
            return;
        }

        Debug.Log("Running AI segmentation...");

        int originalPointCount = _combinedPoints.Count;

        var (predictedLabels, keptIndices) = PointCloudInference.RunModelWithMapping(_combinedPoints);

        if (predictedLabels != null && keptIndices != null)
        {
            Debug.Log($"Inference complete. Original points: {originalPointCount}, Kept after outlier removal: {keptIndices.Length}");

            if (keptIndices.Length > PointCloudInference.NUM_POINTS)
            {
                Debug.Log($"Downsampling {keptIndices.Length} points to {PointCloudInference.NUM_POINTS} for final visualization...");

                var finalPoints = new List<Vector3>(PointCloudInference.NUM_POINTS);
                var finalLabels = new int[PointCloudInference.NUM_POINTS];

                var indicesToSample = Enumerable.Range(0, keptIndices.Length).ToList();

                for (int i = 0; i < indicesToSample.Count; i++)
                {
                    int j = Random.Range(i, indicesToSample.Count);
                    int temp = indicesToSample[i];
                    indicesToSample[i] = indicesToSample[j];
                    indicesToSample[j] = temp;
                }

                for (int i = 0; i < PointCloudInference.NUM_POINTS; i++)
                {
                    int originalIndexInKeptArray = indicesToSample[i];

                    finalPoints.Add(_combinedPoints[keptIndices[originalIndexInKeptArray]]);
                    finalLabels[i] = predictedLabels[originalIndexInKeptArray];
                }

                VisualizeInferenceResult(finalPoints, finalLabels);
            }
            else
            {
                List<Vector3> filteredPoints = new List<Vector3>(keptIndices.Length);
                for (int i = 0; i < keptIndices.Length; i++)
                {
                    filteredPoints.Add(_combinedPoints[keptIndices[i]]);
                }
                VisualizeInferenceResult(filteredPoints, predictedLabels);
            }
        }
        else
        {
            Debug.LogError("Inference failed. Check console for errors from PointCloudInference.");
        }
    }

    // Create a mesh from points using MIConvexHull
    private Mesh CreateMeshFromPoints(List<Vector3> points)
    {
        if (points == null || points.Count < 4)
        {
            Debug.LogWarning("Cannot create a mesh with less than 4 points.");
            return null;
        }

        var vertices = points.Select(p => new DefaultVertex()
        {
            Position = new double[] { p.x, p.y, p.z }
        }).ToList();

        var hull = ConvexHull.Create(vertices);

        if (hull.Outcome != ConvexHullCreationResultOutcome.Success)
        {
            Debug.LogError($"Convex hull creation failed: {hull.Outcome} - {hull.ErrorMessage}");
            return null;
        }

        var hullMesh = new Mesh();
        var hullResult = hull.Result;

        var hullVertices = hullResult.Points.Select(p => new Vector3((float)p.Position[0], (float)p.Position[1], (float)p.Position[2])).ToList();

        var vertexIndexMap = new Dictionary<IVertex, int>();
        for (int i = 0; i < hullResult.Points.Count(); i++)
        {
            vertexIndexMap[hullResult.Points.ElementAt(i)] = i;
        }

        var triangles = new List<int>();
        foreach (var face in hullResult.Faces)
        {
            foreach (var vertex in face.Vertices)
            {
                triangles.Add(vertexIndexMap[vertex]);
            }
        }

        hullMesh.SetVertices(hullVertices);
        hullMesh.SetTriangles(triangles, 0);

        hullMesh.RecalculateNormals();
        hullMesh.RecalculateBounds();

        return hullMesh;
    }

    // Calculate the mesh volume
    public static float CalculateMeshVolume(Mesh mesh)
    {
        if (mesh == null)
        {
            Debug.LogWarning("Mesh is null, cannot calculate volume.");
            return 0f;
        }

        float volume = 0f;
        var verts = mesh.vertices;
        var tris = mesh.triangles;

        for (int i = 0; i < tris.Length; i += 3)
        {
            Vector3 p1 = verts[tris[i]];
            Vector3 p2 = verts[tris[i + 1]];
            Vector3 p3 = verts[tris[i + 2]];
            volume += SignedVolumeOfTriangle(p1, p2, p3);
        }

        return Mathf.Abs(volume);
    }

    // Calculate the signed volume of a triangle
    private static float SignedVolumeOfTriangle(Vector3 p1, Vector3 p2, Vector3 p3)
    {
        return Vector3.Dot(p1, Vector3.Cross(p2, p3)) / 6f;
    }

    // Visualize the inference result
    private void VisualizeInferenceResult(List<Vector3> points, int[] predictedLabels)
    {
        if (predictedLabels.Length != points.Count)
        {
            Debug.LogError($"Mismatch: {points.Count} points but {predictedLabels.Length} predictions!");
            return;
        }

        Debug.Log($"Visualizing {points.Count} points with their corresponding predictions.");

        scanParticles.Clear();

        var particles = new ParticleSystem.Particle[points.Count];

        int chairCount = 0;
        int roomCount = 0;

        var chairPointsForMesh = new List<Vector3>();

        for (int i = 0; i < points.Count; i++)
        {
            particles[i].position = points[i];
            particles[i].startSize = 0.01f;
            particles[i].startLifetime = float.PositiveInfinity;

            if (predictedLabels[i] == 1)
            {
                particles[i].startColor = chairScanColor;
                chairCount++;
                chairPointsForMesh.Add(points[i]);
            }
            else
            {
                particles[i].startColor = roomScanColor;
                roomCount++;
            }
        }

        scanParticles.SetParticles(particles, points.Count);

        Debug.Log($"Visualization complete: {chairCount} chair points (green), {roomCount} room points (white)");

        Mesh chairMesh = CreateMeshFromPoints(chairPointsForMesh);
        if (chairMesh != null)
        {
            float chairVolume = CalculateMeshVolume(chairMesh);
            Debug.Log($"Chair volume: {chairVolume:F4} m³ || Estimated Weight (assuming density of ~50kg/m³): {chairVolume * 50f:F2} kg");

            if (_generatedMeshObject != null) Destroy(_generatedMeshObject);
            _generatedMeshObject = new GameObject("GeneratedChairHull");

            var meshFilter = _generatedMeshObject.AddComponent<MeshFilter>();
            meshFilter.mesh = chairMesh;

            var meshRenderer = _generatedMeshObject.AddComponent<MeshRenderer>();
            meshRenderer.material = generatedMeshMaterial;

            if (meshRenderer.material.color != null)
            {
                Color semiTransparentGreen = chairScanColor;
                semiTransparentGreen.a = 0.3f;
                meshRenderer.material.color = semiTransparentGreen;
            }
        }
    }
}

// Static class to handle saving point cloud data with labels
public static class PointCloudSaver
{
    public static void Save(List<Vector3> pointCloud, List<int> labels, string fileName)
    {
        if (pointCloud == null || pointCloud.Count == 0)
        {
            Debug.LogWarning("Point cloud data is empty. Nothing to save.");
            return;
        }

        string directoryPath = Application.persistentDataPath;
        string filePath = Path.Combine(directoryPath, fileName);

        try
        {
            var sb = new StringBuilder();
            sb.AppendLine("ply");
            sb.AppendLine("format ascii 1.0");
            sb.AppendLine($"element vertex {pointCloud.Count}");
            sb.AppendLine("property float x");
            sb.AppendLine("property float y");
            sb.AppendLine("property float z");
            sb.AppendLine("property uchar label");
            sb.AppendLine("end_header");

            for (int i = 0; i < pointCloud.Count; i++)
            {
                Vector3 point = pointCloud[i];
                sb.AppendLine($"{point.x:F6} {point.y:F6} {point.z:F6} {labels[i]}");
            }

            File.WriteAllText(filePath, sb.ToString());
            Debug.Log($"Point cloud saved successfully with labels to {filePath}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to save point cloud: {ex.Message}");
        }
    }
}

// Point Cloud Inference Setup and Execution
public static class PointCloudInference
{
    private static Worker m_Worker;
    private static bool s_IsInitialized = false;
    public static int NUM_POINTS = 16384;

    // Initialize Sentis ready to run
    public static void Initialize(ModelAsset modelAsset)
    {
        if (s_IsInitialized) return;

        var model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);

        s_IsInitialized = true;
        Debug.Log("PointCloudInference initialized successfully.");
    }

    // Cleanup to prevent memory leaks when closing program
    public static void Shutdown()
    {
        m_Worker?.Dispose();
        s_IsInitialized = false;
    }

    // Run the model with mapping to keep track of points
    public static (int[] predictedLabels, int[] keptPointIndices) RunModelWithMapping(List<Vector3> rawPoints)
    {
        if (!s_IsInitialized)
        {
            Debug.LogError("Inference Engine not initialized! Call PointCloudInference.Initialize() first.");
            return (null, null);
        }

        if (rawPoints == null || rawPoints.Count == 0)
        {
            Debug.LogWarning("Input point cloud is empty. Aborting inference.");
            return (null, null);
        }

        var (modelInput, pointMapping, keptIndices) = PreprocessDataWithMapping(rawPoints);

        var inputTensor = new Tensor<float>(new TensorShape(1, 3, NUM_POINTS), modelInput);

        m_Worker.Schedule(inputTensor);

        var outputTensor = m_Worker.PeekOutput().ReadbackAndClone() as Tensor<float>;

        int[] modelPredictions = GetPredictedLabels(outputTensor);

        int[] keptPointLabels = MapPredictionsToKeptPoints(keptIndices.Length, modelPredictions, pointMapping, keptIndices);

        inputTensor.Dispose();
        outputTensor.Dispose();

        Debug.Log($"Inference complete. Original points: {rawPoints.Count}, Kept after outlier removal: {keptIndices.Length}, Model predictions: {modelPredictions.Length}");

        return (keptPointLabels, keptIndices);
    }

    // Process the data with the mapping
    private static (float[] modelInput, int[] pointMapping, int[] keptIndices) PreprocessDataWithMapping(List<Vector3> points)
    {
        List<Vector3> convertedPoints = new List<Vector3>(points.Count);
        foreach (var p in points)
        {
            convertedPoints.Add(new Vector3(p.x, p.y, -p.z));
        }

        var filteredPoints = new List<Vector3>(convertedPoints);
        int[] originalIndices = Enumerable.Range(0, filteredPoints.Count).ToArray();

        if (filteredPoints.Count == 0)
        {
            Debug.LogWarning("Point cloud is empty after conversion/filtering.");
            return (new float[NUM_POINTS * 3], new int[NUM_POINTS], new int[0]);
        }

        Vector3[] sampledPointsArray = new Vector3[NUM_POINTS];
        int[] pointMapping = new int[NUM_POINTS];
        if (filteredPoints.Count >= NUM_POINTS)
        {
            var indicesToSample = Enumerable.Range(0, filteredPoints.Count).ToList();
            for (int i = 0; i < indicesToSample.Count; i++)
            {
                int j = Random.Range(i, indicesToSample.Count);
                (indicesToSample[i], indicesToSample[j]) = (indicesToSample[j], indicesToSample[i]);
            }

            for (int i = 0; i < NUM_POINTS; i++)
            {
                int index = indicesToSample[i];
                sampledPointsArray[i] = filteredPoints[index];
                pointMapping[i] = originalIndices[index];
            }
        }
        else
        {
            for (int i = 0; i < filteredPoints.Count; i++)
            {
                sampledPointsArray[i] = filteredPoints[i];
                pointMapping[i] = originalIndices[i];
            }
            for (int i = filteredPoints.Count; i < NUM_POINTS; i++)
            {
                int randomIndex = Random.Range(0, filteredPoints.Count);
                sampledPointsArray[i] = filteredPoints[randomIndex];
                pointMapping[i] = originalIndices[randomIndex];
            }
        }

        List<Vector3> sampledPoints = new List<Vector3>(sampledPointsArray);

        Vector3 center = Vector3.zero;
        foreach (var p in sampledPoints) center += p;
        center /= sampledPoints.Count;

        for (int i = 0; i < sampledPoints.Count; i++)
        {
            sampledPoints[i] -= center;
        }

        float maxDist = 0f;
        foreach (var p in sampledPoints) maxDist = Mathf.Max(maxDist, p.magnitude);
        if (maxDist > 1e-6f)
        {
            for (int i = 0; i < sampledPoints.Count; i++) sampledPoints[i] /= maxDist;
        }

        float[] floatData = new float[NUM_POINTS * 3];
        for (int i = 0; i < NUM_POINTS; i++)
        {
            floatData[i] = sampledPoints[i].x;
            floatData[i + NUM_POINTS] = sampledPoints[i].y;
            floatData[i + 2 * NUM_POINTS] = sampledPoints[i].z;
        }

        return (floatData, pointMapping, pointMapping);
    }

    // Using mapped points find the kept points
    private static int[] MapPredictionsToKeptPoints(int keptPointCount, int[] modelPredictions, int[] pointMapping, int[] keptIndices)
    {
        int[] keptPointLabels = new int[keptPointCount];
        int[] voteCounts = new int[keptPointCount * 2];

        for (int modelIdx = 0; modelIdx < modelPredictions.Length; modelIdx++)
        {
            int originalIdx = pointMapping[modelIdx];
            int prediction = modelPredictions[modelIdx];

            int keptIdx = -1;
            for (int k = 0; k < keptIndices.Length; k++)
            {
                if (keptIndices[k] == originalIdx)
                {
                    keptIdx = k;
                    break;
                }
            }

            if (keptIdx >= 0)
            {
                voteCounts[keptIdx * 2 + prediction]++;
            }
        }

        for (int i = 0; i < keptPointCount; i++)
        {
            int votesForRoom = voteCounts[i * 2 + 0];
            int votesForChair = voteCounts[i * 2 + 1];

            keptPointLabels[i] = (votesForChair > votesForRoom) ? 1 : 0;
        }

        return keptPointLabels;
    }

    // Get the predicited labels from each point based on score
    private static int[] GetPredictedLabels(Tensor<float> outputTensor)
    {
        int[] labels = new int[NUM_POINTS];
        for (int i = 0; i < NUM_POINTS; i++)
        {
            float scoreRoom = outputTensor[0, i, 0];
            float scoreChair = outputTensor[0, i, 1];
            labels[i] = (scoreChair > scoreRoom) ? 1 : 0;
        }
        return labels;
    }
}