package org.mapreduce.mapreduce;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Reducer pass-through: nếu có nhiều giá trị cho cùng file (hiếm vì mỗi file là một record),
 * ta vẫn ghi tất cả (SequenceFile cho phép key trùng).
 * Nếu bạn muốn ghi thành file riêng cho mỗi ảnh, cần custom OutputFormat (tự tạo).
 */
public class ImageReducer extends Reducer<Text, BytesWritable, Text, BytesWritable> {
    @Override
    protected void reduce(Text key, Iterable<BytesWritable> values, Context context)
            throws IOException, InterruptedException {
        for (BytesWritable v : values) {
            // emit filename -> processed bytes
            context.write(key, v);
        }
    }
}

