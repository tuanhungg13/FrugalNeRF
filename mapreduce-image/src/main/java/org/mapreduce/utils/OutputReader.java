package org.mapreduce.utils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import java.io.IOException;

/**
 * Utility to read and display MapReduce output
 */
public class OutputReader {
    
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("Usage: OutputReader <output_path>");
            System.exit(1);
        }
        
        String outputPath = args[0];
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path path = new Path(outputPath);
        
        System.out.println("Reading output from: " + outputPath);
        System.out.println("=====================================");
        
        try (SequenceFile.Reader reader = new SequenceFile.Reader(conf, 
                SequenceFile.Reader.file(path))) {
            
            Text key = new Text();
            BytesWritable value = new BytesWritable();
            
            int recordCount = 0;
            while (reader.next(key, value)) {
                recordCount++;
                System.out.println("Record " + recordCount + ":");
                System.out.println("  Key: " + key.toString());
                System.out.println("  Value size: " + value.getLength() + " bytes");
                System.out.println("  Value type: " + (value.getLength() > 0 ? "Binary data" : "Empty"));
                System.out.println("---");
            }
            
            System.out.println("Total records: " + recordCount);
        }
    }
}

