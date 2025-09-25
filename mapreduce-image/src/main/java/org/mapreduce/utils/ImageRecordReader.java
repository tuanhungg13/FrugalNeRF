package org.mapreduce.utils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class ImageRecordReader extends RecordReader<Text, BytesWritable> {
    private FileSplit fileSplit;
    private boolean processed = false;
    private Text currentKey = new Text();
    private BytesWritable currentValue = new BytesWritable();

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException {
        this.fileSplit = (FileSplit) split;
    }

    @Override
    public boolean nextKeyValue() throws IOException {
        if (processed) return false;

        Path path = fileSplit.getPath();
        Configuration conf = fileSplit.getPath().getFileSystem(new org.apache.hadoop.conf.Configuration()).getConf();
        FileSystem fs = path.getFileSystem(conf);

        FSDataInputStream in = null;
        try {
            in = fs.open(path);
            long len = fs.getFileStatus(path).getLen();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int read;
            while ((read = in.read(buf)) > 0) {
                baos.write(buf, 0, read);
            }
            byte[] bytes = baos.toByteArray();
            currentKey.set(path.getName());
            currentValue.set(bytes, 0, bytes.length);
            processed = true;
            return true;
        } finally {
            if (in != null) in.close();
        }
    }

    @Override
    public Text getCurrentKey() { return currentKey; }

    @Override
    public BytesWritable getCurrentValue() { return currentValue; }

    @Override
    public float getProgress() { return processed ? 1.0f : 0.0f; }

    @Override
    public void close() {}
}

