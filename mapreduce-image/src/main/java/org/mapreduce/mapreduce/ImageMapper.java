package org.mapreduce.mapreduce;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.mapreduce.utils.ImageUtils;

import java.awt.image.BufferedImage;
import java.io.IOException;

public class ImageMapper extends Mapper<Text, BytesWritable, Text, BytesWritable> {

    // target size — bạn có thể param hóa qua Configuration nếu muốn
    private static final int TARGET_W = 256;
    private static final int TARGET_H = 256;
    private static final boolean TO_GRAY = true;

    @Override
    protected void map(Text key, BytesWritable value, Context context) throws IOException, InterruptedException {
        String filename = key.toString();
        byte[] imgBytes = value.copyBytes();

        try {
            BufferedImage img = ImageUtils.bytesToBufferedImage(imgBytes);
            if (img == null) return;

            // Resize
            BufferedImage resized = ImageUtils.resize(img, TARGET_W, TARGET_H);

            // Grayscale (tùy chọn)
            BufferedImage processed = TO_GRAY ? ImageUtils.toGray(resized) : resized;

            // Convert back to bytes (preserve format based on filename)
            String format = ImageUtils.detectFormatFromName(filename);
            byte[] outBytes = ImageUtils.bufferedImageToBytes(processed, format);
            context.getCounter("PROCESSED", format.toUpperCase()).increment(1);
            context.write(new Text(filename), new BytesWritable(outBytes));
        } catch (Exception e) {
            // log and skip this image
            System.err.println("Error processing " + filename + ": " + e.getMessage());
        }
    }
}
