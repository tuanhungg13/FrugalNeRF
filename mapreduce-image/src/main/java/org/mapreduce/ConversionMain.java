package org.mapreduce;

import org.mapreduce.utils.FrugalNeRFConverter;

/**
 * Main class để convert MapReduce output sang FrugalNeRF format
 */
public class ConversionMain {
    
    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: ConversionMain <mapreduce_output_dir> <frugalnerf_output_dir>");
            System.exit(-1);
        }
        
        String mapreduceOutputDir = args[0];
        String frugalnerfOutputDir = args[1];
        
        System.out.println("Starting MapReduce to FrugalNeRF conversion...");
        System.out.println("MapReduce output: " + mapreduceOutputDir);
        System.out.println("FrugalNeRF output: " + frugalnerfOutputDir);
        
        try {
            // Convert MapReduce output to FrugalNeRF format
            FrugalNeRFConverter.convertToFrugalNeRF(mapreduceOutputDir, frugalnerfOutputDir);
            
            System.out.println("Conversion completed successfully!");
            System.out.println("FrugalNeRF dataset ready at: " + frugalnerfOutputDir);
            
        } catch (Exception e) {
            System.err.println("Conversion failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(-1);
        }
    }
}
