package org.mapreduce.utils;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Converter từ MapReduce output sang FrugalNeRF dataset format
 * Tạo poses_bounds.npy, images/, và metadata files
 */
public class FrugalNeRFConverter {
    
    private static final Pattern IMAGE_PATTERN = Pattern.compile(".*\\.(jpg|jpeg|png|bmp|tiff|tif)$", Pattern.CASE_INSENSITIVE);
    
    /**
     * Convert MapReduce output to FrugalNeRF format
     */
    public static void convertToFrugalNeRF(String mapreduceOutputDir, String frugalnerfOutputDir) {
        try {
            System.out.println("Converting MapReduce output to FrugalNeRF format...");
            System.out.println("Input: " + mapreduceOutputDir);
            System.out.println("Output: " + frugalnerfOutputDir);
            
            // Create output directory
            Files.createDirectories(Paths.get(frugalnerfOutputDir));
            Files.createDirectories(Paths.get(frugalnerfOutputDir, "images"));
            
            // Parse MapReduce output files (SequenceFiles under output directory)
            List<SceneData> scenes = parseMapReduceOutput(mapreduceOutputDir);
            
            for (SceneData scene : scenes) {
                System.out.println("Processing scene: " + scene.sceneId);
                
                // Create scene directory
                String sceneDir = frugalnerfOutputDir + "/" + scene.sceneId;
                Files.createDirectories(Paths.get(sceneDir));
                Files.createDirectories(Paths.get(sceneDir, "images"));
                
                // Convert scene data
                convertScene(scene, sceneDir);
            }
            
            System.out.println("Conversion completed successfully!");
            
        } catch (Exception e) {
            System.err.println("Error converting to FrugalNeRF format: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Parse MapReduce output files
     */
    private static List<SceneData> parseMapReduceOutput(String outputDir) {
        List<SceneData> scenes = new ArrayList<>();
        
        try {
            // Expect part-r-xxxxx SequenceFiles inside outputDir
            Files.list(Paths.get(outputDir))
                .filter(Files::isRegularFile)
                .filter(path -> path.getFileName().toString().startsWith("part-"))
                .forEach(path -> readScenesFromSequenceFile(path.toString(), scenes));
                
        } catch (IOException e) {
            System.err.println("Error reading MapReduce output: " + e.getMessage());
        }
        
        return scenes;
    }
    
    /**
     * Parse individual scene file
     */
    private static void readScenesFromSequenceFile(String filePath, List<SceneData> outScenes) {
        try {
            org.apache.hadoop.conf.Configuration conf = new org.apache.hadoop.conf.Configuration();
            // Ensure local files are read via LocalFileSystem
            conf.set("fs.defaultFS", "file:///");
            java.net.URI uri = new java.io.File(filePath).toURI();
            org.apache.hadoop.fs.Path path = new org.apache.hadoop.fs.Path(uri);
            org.apache.hadoop.io.SequenceFile.Reader reader = new org.apache.hadoop.io.SequenceFile.Reader(conf,
                org.apache.hadoop.io.SequenceFile.Reader.file(path));
            org.apache.hadoop.io.Text key = new org.apache.hadoop.io.Text();
            org.apache.hadoop.io.BytesWritable value = new org.apache.hadoop.io.BytesWritable();
            while (reader.next(key, value)) {
                byte[] bytes = value.copyBytes();
                SceneData scene = parseSceneFromBytes(bytes, value.getLength());
                if (scene != null) outScenes.add(scene);
            }
            reader.close();
        } catch (Exception e) {
            System.err.println("Error reading SequenceFile " + filePath + ": " + e.getMessage());
        }
    }

    private static SceneData parseSceneFromBytes(byte[] data, int len) {
        try {
            // Stream-like parsing over byte[] to reduce peak memory
            SceneData scene = new SceneData();
            java.util.ArrayList<float[]> posesRows = new java.util.ArrayList<>();
            java.util.ArrayList<float[]> boundsRows = new java.util.ArrayList<>();
            java.util.ArrayList<ProcessedImageData> images = new java.util.ArrayList<>();
            boolean inPoses = false, inIntrinsics = false, inImages = false;
            int intrRow = 0;
            float[][][] intr = new float[1][3][3];
            ProcessedImageData curImg = null;

            int i = 0; StringBuilder sb = new StringBuilder(256);
            while (i < len) {
                sb.setLength(0);
                // read one line (up to \n)
                while (i < len) {
                    byte b = data[i++];
                    if (b == '\n') break;
                    sb.append((char)(b & 0xFF));
                }
                String line = sb.toString().trim();
                if (line.isEmpty()) {
                    // end sections on blank
                    if (inPoses) inPoses = false;
                    if (inIntrinsics) inIntrinsics = false;
                    continue;
                }
                if (line.startsWith("SCENE_ID:")) {
                    scene.sceneId = line.substring(9).trim();
                    continue;
                }
                if (line.startsWith("NUM_IMAGES:")) {
                    scene.numImages = Integer.parseInt(line.substring(11).trim());
                    continue;
                }
                if (line.startsWith("WHITE_BG:")) {
                    scene.whiteBg = Boolean.parseBoolean(line.substring(9).trim());
                    continue;
                }
                if (line.startsWith("SCENE_BBOX:")) {
                    String[] p = line.substring(11).split(",");
                    scene.sceneBbox = new float[2][3];
                    scene.sceneBbox[0][0] = Float.parseFloat(p[0]);
                    scene.sceneBbox[0][1] = Float.parseFloat(p[1]);
                    scene.sceneBbox[0][2] = Float.parseFloat(p[2]);
                    scene.sceneBbox[1][0] = Float.parseFloat(p[3]);
                    scene.sceneBbox[1][1] = Float.parseFloat(p[4]);
                    scene.sceneBbox[1][2] = Float.parseFloat(p[5]);
                    continue;
                }
                if (line.startsWith("NEAR_FAR:")) {
                    String[] p = line.substring(9).split(",");
                    scene.nearFar = new float[]{Float.parseFloat(p[0]), Float.parseFloat(p[1])};
                    continue;
                }
                if (line.equals("POSES_BOUNDS:")) { inPoses = true; continue; }
                if (line.equals("INTRINSICS:")) { inIntrinsics = true; intrRow = 0; continue; }
                if (line.equals("IMAGES:")) { inImages = true; continue; }

                if (inPoses) {
                    String[] parts = line.split("\\s+");
                    if (parts.length >= 17) {
                        float[] row = new float[17];
                        for (int k = 0; k < 17; k++) row[k] = Float.parseFloat(parts[k]);
                        posesRows.add(row);
                        boundsRows.add(new float[]{row[15], row[16]});
                    }
                    continue;
                }
                if (inIntrinsics && intrRow < 3) {
                    String[] parts = line.split("\\s+");
                    if (parts.length >= 3) {
                        for (int c = 0; c < 3; c++) intr[0][intrRow][c] = Float.parseFloat(parts[c]);
                        intrRow++;
                    }
                    continue;
                }
                if (inImages) {
                    if (line.startsWith("IMAGE_")) {
                        if (curImg != null) images.add(curImg);
                        curImg = new ProcessedImageData();
                        continue;
                    }
                    if (line.startsWith("FILENAME:")) { if (curImg != null) curImg.filename = line.substring(9).trim(); continue; }
                    if (line.startsWith("WIDTH:")) { if (curImg != null) curImg.width = Integer.parseInt(line.substring(6).trim()); continue; }
                    if (line.startsWith("HEIGHT:")) { if (curImg != null) curImg.height = Integer.parseInt(line.substring(7).trim()); continue; }
                }
            }
            if (curImg != null) images.add(curImg);

            int n = posesRows.size();
            scene.poses = new float[n][3][5];
            for (int r = 0; r < n; r++) {
                float[] row = posesRows.get(r);
                int idx = 0;
                for (int rr = 0; rr < 3; rr++) {
                    for (int cc = 0; cc < 5; cc++) scene.poses[r][rr][cc] = row[idx++];
                }
            }
            scene.bounds = new float[boundsRows.size()][2];
            for (int r = 0; r < boundsRows.size(); r++) scene.bounds[r] = boundsRows.get(r);
            scene.intrinsics = intr;
            scene.images = images;
            return scene;
        } catch (Exception e) {
            System.err.println("Error parsing scene bytes: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Convert scene to FrugalNeRF format
     */
    private static void convertScene(SceneData scene, String sceneDir) {
        try {
            // 1. Create poses_bounds.npy
            createPosesBoundsFile(scene, sceneDir + "/poses_bounds.npy");
            
            // 2. Copy images
            copyImages(scene, sceneDir + "/images");
            
            // NOTE: For FrugalNeRF training, only poses_bounds.npy and images/ are required.
            // Skipping metadata and split generation to keep output minimal.
            
        } catch (Exception e) {
            System.err.println("Error converting scene " + scene.sceneId + ": " + e.getMessage());
        }
    }
    
    /**
     * Create poses_bounds.npy file
     */
    private static void createPosesBoundsFile(SceneData scene, String outputPath) {
        try {
            // Create poses_bounds array (N x 17)
            float[][] posesBounds = new float[scene.numImages][17];
            
            for (int i = 0; i < scene.numImages; i++) {
                // Flatten pose matrix (3x5 -> 15 values)
                int idx = 0;
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 5; k++) {
                        posesBounds[i][idx++] = scene.poses[i][j][k];
                    }
                }
                // Add bounds (2 values)
                posesBounds[i][15] = scene.bounds[i][0]; // near
                posesBounds[i][16] = scene.bounds[i][1]; // far
            }
            
            // Write to file as true NumPy .npy float32 array
            writeNpyFloat2D(posesBounds, outputPath);
            
        } catch (Exception e) {
            System.err.println("Error creating poses_bounds.npy: " + e.getMessage());
        }
    }
    
    /**
     * Copy images to output directory
     */
    private static void copyImages(SceneData scene, String imagesDir) {
        try {
            for (int i = 0; i < scene.images.size(); i++) {
                ProcessedImageData img = scene.images.get(i);
                String outputPath = imagesDir + "/" + String.format("IMG_%04d.jpg", i);
                String src = img.filename;
                if (src == null || src.isEmpty()) {
                    // skip if no source
                    continue;
                }
                if (src.startsWith("hdfs://")) {
                    // Copy from HDFS to local
                    org.apache.hadoop.conf.Configuration conf = new org.apache.hadoop.conf.Configuration();
                    org.apache.hadoop.fs.Path hdfsPath = new org.apache.hadoop.fs.Path(src);
                    org.apache.hadoop.fs.FileSystem fs = org.apache.hadoop.fs.FileSystem.get(hdfsPath.toUri(), conf);
                    try (java.io.InputStream in = fs.open(hdfsPath)) {
                        Files.copy(in, Paths.get(outputPath), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                    }
                } else {
                    // Local/absolute path
                    Files.copy(Paths.get(src), Paths.get(outputPath), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                }
            }
            
        } catch (Exception e) {
            System.err.println("Error copying images: " + e.getMessage());
        }
    }
    
    /**
     * Create metadata files
     */
    private static void createMetadataFiles(SceneData scene, String sceneDir) {
        try {
            // Create scene metadata JSON
            String metadata = createSceneMetadata(scene);
            Files.write(Paths.get(sceneDir + "/metadata.json"), metadata.getBytes());
            
            // Create intrinsics file
            String intrinsics = createIntrinsicsData(scene);
            Files.write(Paths.get(sceneDir + "/intrinsics.txt"), intrinsics.getBytes());
            
        } catch (Exception e) {
            System.err.println("Error creating metadata files: " + e.getMessage());
        }
    }
    
    /**
     * Create train/test split
     */
    private static void createTrainTestSplit(SceneData scene, String sceneDir) {
        try {
            // Create train/test split (80% train, 20% test)
            int totalImages = scene.numImages;
            int trainSize = (int) (totalImages * 0.8);
            
            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> testIndices = new ArrayList<>();
            
            for (int i = 0; i < totalImages; i++) {
                if (i < trainSize) {
                    trainIndices.add(i);
                } else {
                    testIndices.add(i);
                }
            }
            
            // Write split files
            writeSplitFile(sceneDir + "/train_indices.txt", trainIndices);
            writeSplitFile(sceneDir + "/test_indices.txt", testIndices);
            
        } catch (Exception e) {
            System.err.println("Error creating train/test split: " + e.getMessage());
        }
    }
    
    // Helper methods
    
    private static String extractValue(String content, String key) {
        String pattern = key + ":([^\\n]+)";
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(pattern);
        java.util.regex.Matcher m = p.matcher(content);
        
        if (m.find()) {
            return m.group(1).trim();
        }
        
        return "";
    }
    
    private static float[][][] parsePosesSection(String content) {
        // Parse after line "POSES_BOUNDS:" then read N lines, each has 17 floats: 15 pose + 2 bounds
        List<float[]> rows = new ArrayList<>();
        String[] lines = content.split("\n");
        boolean inPoses = false;
        for (String line : lines) {
            if (line.trim().equals("POSES_BOUNDS:")) { inPoses = true; continue; }
            if (!inPoses) continue;
            if (line.trim().isEmpty()) break;
            String[] parts = line.trim().split("\\s+");
            if (parts.length < 17) break;
            float[] vals = new float[17];
            for (int i = 0; i < 17 && i < parts.length; i++) vals[i] = Float.parseFloat(parts[i]);
            rows.add(vals);
        }
        int n = rows.size();
        float[][][] poses = new float[n][3][5];
        for (int i = 0; i < n; i++) {
            float[] row = rows.get(i);
            int idx = 0;
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 5; c++) {
                    poses[i][r][c] = row[idx++];
                }
            }
        }
        return poses;
    }
    
    private static float[][] parseBoundsSection(String content, int expected, float[] nearFarDefault) {
        List<float[]> rows = new ArrayList<>();
        String[] lines = content.split("\n");
        boolean inPoses = false;
        for (String line : lines) {
            if (line.trim().equals("POSES_BOUNDS:")) { inPoses = true; continue; }
            if (!inPoses) continue;
            if (line.trim().isEmpty()) break;
            String[] parts = line.trim().split("\\s+");
            if (parts.length < 17) break;
            float near = Float.parseFloat(parts[15]);
            float far = Float.parseFloat(parts[16]);
            rows.add(new float[]{near, far});
        }
        if (rows.isEmpty() && expected > 0) {
            rows = new ArrayList<>();
            for (int i = 0; i < expected; i++) rows.add(new float[]{nearFarDefault[0], nearFarDefault[1]});
        }
        float[][] bounds = new float[rows.size()][2];
        for (int i = 0; i < rows.size(); i++) bounds[i] = rows.get(i);
        return bounds;
    }
    
    private static float[][][] parseIntrinsicsSection(String content) {
        float[][][] intr = new float[1][3][3];
        String[] lines = content.split("\n");
        boolean inIntr = false;
        int r = 0;
        for (String line : lines) {
            if (line.trim().equals("INTRINSICS:")) { inIntr = true; continue; }
            if (!inIntr) continue;
            if (line.trim().isEmpty()) break;
            String[] parts = line.trim().split("\\s+");
            if (parts.length < 3) break;
            for (int c = 0; c < 3; c++) intr[0][r][c] = Float.parseFloat(parts[c]);
            r++; if (r >= 3) break;
        }
        return intr;
    }
    
    private static float[][][] parseDirections(String content) {
        // Parse directions from content
        return new float[256][256][3]; // Placeholder
    }
    
    private static List<ProcessedImageData> parseImagesSection(String content) {
        List<ProcessedImageData> images = new ArrayList<>();
        String[] lines = content.split("\n");
        ProcessedImageData cur = null;
        for (String line : lines) {
            String t = line.trim();
            if (t.startsWith("IMAGE_")) {
                if (cur != null) images.add(cur);
                cur = new ProcessedImageData();
            } else if (t.startsWith("FILENAME:")) {
                if (cur != null) cur.filename = t.substring(9);
            } else if (t.startsWith("WIDTH:")) {
                if (cur != null) cur.width = Integer.parseInt(t.substring(6));
            } else if (t.startsWith("HEIGHT:")) {
                if (cur != null) cur.height = Integer.parseInt(t.substring(7));
            }
        }
        if (cur != null) images.add(cur);
        return images;
    }
    
    private static void writeNpyFloat2D(float[][] data, String outputPath) {
        // Minimal .npy (v1.0) writer for 2D float32 C-order arrays
        try (java.io.FileOutputStream fos = new java.io.FileOutputStream(outputPath)) {
            // Magic string + version
            fos.write(new byte[]{(byte)0x93,'N','U','M','P','Y',1,0});
            int rows = data.length;
            int cols = rows > 0 ? data[0].length : 0;
            String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + rows + ", " + cols + "), }";
            // Pad header to 16-byte alignment including 10-byte magic/version+2-byte header-len
            int headerLen = header.getBytes(java.nio.charset.StandardCharsets.US_ASCII).length + 1; // newline
            int pad = 16 - ((10 + 2 + headerLen) % 16);
            if (pad == 16) pad = 0;
            StringBuilder headerPadded = new StringBuilder(header);
            for (int i = 0; i < pad; i++) headerPadded.append(' ');
            headerPadded.append('\n');
            byte[] headerBytes = headerPadded.toString().getBytes(java.nio.charset.StandardCharsets.US_ASCII);
            // Write little-endian uint16 header length
            int hlen = headerBytes.length;
            fos.write(new byte[]{(byte)(hlen & 0xFF), (byte)((hlen >> 8) & 0xFF)});
            fos.write(headerBytes);
            // Write data row-major, float32 LE
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(rows * cols * 4);
            bb.order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < rows; i++) {
                float[] row = data[i];
                for (int j = 0; j < cols; j++) bb.putFloat(row[j]);
            }
            fos.write(bb.array());
        } catch (Exception e) {
            System.err.println("Error writing .npy file: " + e.getMessage());
        }
    }
    
    private static String createSceneMetadata(SceneData scene) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"scene_id\": \"").append(scene.sceneId).append("\",\n");
        sb.append("  \"num_images\": ").append(scene.numImages).append(",\n");
        sb.append("  \"scene_bbox\": [\n");
        sb.append("    [").append(scene.sceneBbox[0][0]).append(", ").append(scene.sceneBbox[0][1]).append(", ").append(scene.sceneBbox[0][2]).append("],\n");
        sb.append("    [").append(scene.sceneBbox[1][0]).append(", ").append(scene.sceneBbox[1][1]).append(", ").append(scene.sceneBbox[1][2]).append("]\n");
        sb.append("  ],\n");
        sb.append("  \"near_far\": [").append(scene.nearFar[0]).append(", ").append(scene.nearFar[1]).append("],\n");
        sb.append("  \"white_bg\": ").append(scene.whiteBg).append("\n");
        sb.append("}\n");
        return sb.toString();
    }
    
    private static String createIntrinsicsData(SceneData scene) {
        StringBuilder sb = new StringBuilder();
        float[][] intrinsic = scene.intrinsics[0];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sb.append(intrinsic[i][j]).append(" ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
    
    private static void writeSplitFile(String outputPath, List<Integer> indices) {
        try {
            StringBuilder sb = new StringBuilder();
            for (int idx : indices) {
                sb.append(idx).append("\n");
            }
            Files.write(Paths.get(outputPath), sb.toString().getBytes());
            
        } catch (Exception e) {
            System.err.println("Error writing split file: " + e.getMessage());
        }
    }
    
    /**
     * Data structure for scene
     */
    private static class SceneData {
        String sceneId;
        int numImages;
        float[][] sceneBbox;
        float[] nearFar;
        float[][][] poses;
        float[][] bounds;
        float[][][] intrinsics;
        float[][][] directions;
        List<ProcessedImageData> images;
        boolean whiteBg;
    }
    
    /**
     * Data structure for processed image
     */
    private static class ProcessedImageData {
        String filename;
        int width, height;
        float[][] depthMap;
        float[][][] rays;
    }
}
