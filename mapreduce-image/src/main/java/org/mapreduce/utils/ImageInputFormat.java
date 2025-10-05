package org.mapreduce.utils;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import java.io.IOException;

public class ImageInputFormat extends FileInputFormat<Text, BytesWritable> {
    
    public static class ImagePathFilter implements PathFilter {
        @Override
        public boolean accept(Path path) {
            String name = path.getName().toLowerCase();
            return name.endsWith(".jpg") || name.endsWith(".jpeg") || 
                   name.endsWith(".png") || name.endsWith(".bmp") || 
                   name.endsWith(".tiff") || name.endsWith(".tif");
        }
    }
    
    @Override
    protected boolean isSplitable(JobContext context, Path filename) {
        return false; // mỗi file ảnh = 1 record
    }

    @Override
    public RecordReader<Text, BytesWritable> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException, InterruptedException {
        return new ImageRecordReader();
    }
}

