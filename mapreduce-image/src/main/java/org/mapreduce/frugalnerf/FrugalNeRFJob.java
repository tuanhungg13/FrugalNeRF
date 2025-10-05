package org.mapreduce.frugalnerf;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.mapreduce.utils.ImageInputFormat;

/**
 * FrugalNeRF Data Preprocessing Job
 * Xử lý dữ liệu ảnh cho FrugalNeRF training
 */
public class FrugalNeRFJob {
    
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FrugalNeRFJob <input_hdfs_dir> <output_hdfs_dir> [options]");
            System.err.println("Options:");
            System.err.println("  --target-width WIDTH     Target image width (default: 256)");
            System.err.println("  --target-height HEIGHT   Target image height (default: 256)");
            System.err.println("  --estimate-depth         Enable depth estimation (default: true)");
            System.err.println("  --generate-rays          Enable ray generation (default: true)");
            System.exit(-1);
        }
        
        String inputDir = args[0];
        String outputDir = args[1];
        
        // Parse additional options
        int targetWidth = 256;
        int targetHeight = 256;
        boolean estimateDepth = true;
        boolean generateRays = true;
        
        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--target-width":
                    if (i + 1 < args.length) {
                        targetWidth = Integer.parseInt(args[++i]);
                    }
                    break;
                case "--target-height":
                    if (i + 1 < args.length) {
                        targetHeight = Integer.parseInt(args[++i]);
                    }
                    break;
                case "--estimate-depth":
                    estimateDepth = true;
                    break;
                case "--no-estimate-depth":
                    estimateDepth = false;
                    break;
                case "--generate-rays":
                    generateRays = true;
                    break;
                case "--no-generate-rays":
                    generateRays = false;
                    break;
            }
        }
        
        // Create configuration
        Configuration conf = new Configuration();
        conf.setInt("frugalnerf.target.width", targetWidth);
        conf.setInt("frugalnerf.target.height", targetHeight);
        conf.setBoolean("frugalnerf.estimate.depth", estimateDepth);
        conf.setBoolean("frugalnerf.generate.rays", generateRays);
        
        // Create job
        Job job = Job.getInstance(conf, "FrugalNeRF Data Preprocessing");
        job.setJarByClass(FrugalNeRFJob.class);
        
        // Set input format
        job.setInputFormatClass(ImageInputFormat.class);
        ImageInputFormat.addInputPath(job, new Path(inputDir));
        
        // Set path filter to only process image files
        // ImageInputFormat.setInputPathFilter(job, ImageInputFormat.ImagePathFilter.class);
        
        // Set mapper and reducer
        job.setMapperClass(FrugalNeRFMapper.class);
        job.setReducerClass(FrugalNeRFReducer.class);
        
        // Set output classes
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(BytesWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(BytesWritable.class);
        
        // Set output format
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        SequenceFileOutputFormat.setOutputPath(job, new Path(outputDir));
        
        // Set number of reducers
        job.setNumReduceTasks(4);
        
        // Print job configuration
        System.out.println("FrugalNeRF Job Configuration:");
        System.out.println("  Input Directory: " + inputDir);
        System.out.println("  Output Directory: " + outputDir);
        System.out.println("  Target Size: " + targetWidth + "x" + targetHeight);
        System.out.println("  Estimate Depth: " + estimateDepth);
        System.out.println("  Generate Rays: " + generateRays);
        
        // Submit job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
