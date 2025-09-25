package org.mapreduce.utils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class ImageUtils {

    // bytes -> BufferedImage
    public static BufferedImage bytesToBufferedImage(byte[] data) throws IOException {
        if (data == null) return null;
        try (ByteArrayInputStream bis = new ByteArrayInputStream(data)) {
            return ImageIO.read(bis);
        }
    }

    // BufferedImage -> bytes (format: "jpg" or "png")
    public static byte[] bufferedImageToBytes(BufferedImage img, String format) throws IOException {
        if (img == null) return null;
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            ImageIO.write(img, format, baos);
            return baos.toByteArray();
        }
    }

    // Resize image to targetW x targetH (preserve aspect by scaling to fit then center-crop optional)
    public static BufferedImage resize(BufferedImage src, int targetW, int targetH) {
        Image scaled = src.getScaledInstance(targetW, targetH, Image.SCALE_SMOOTH);
        BufferedImage out = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = out.createGraphics();
        g2d.setComposite(AlphaComposite.Src);
        g2d.drawImage(scaled, 0, 0, null);
        g2d.dispose();
        return out;
    }

    // Convert to grayscale
    public static BufferedImage toGray(BufferedImage src) {
        BufferedImage gray = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = gray.getGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return gray;
    }

    // Try to detect format from name (fallback jpg)
    public static String detectFormatFromName(String name) {
        if (name == null) return "jpg";
        String lower = name.toLowerCase();
        if (lower.endsWith(".png")) return "png";
        if (lower.endsWith(".bmp")) return "bmp";
        if (lower.endsWith(".gif")) return "gif";
        return "jpg";
    }
}
