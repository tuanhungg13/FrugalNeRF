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

    // Convert to sRGB 8-bit RGB
    public static BufferedImage toSRGB(BufferedImage src) {
        if (src == null) return null;
        if (src.getType() == BufferedImage.TYPE_INT_RGB) return src;
        BufferedImage out = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = out.createGraphics();
        g2d.setComposite(AlphaComposite.Src);
        g2d.drawImage(src, 0, 0, null);
        g2d.dispose();
        return out;
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

    // Center-crop to target aspect before resize
    public static BufferedImage centerCropToAspect(BufferedImage src, int targetW, int targetH) {
        double targetAspect = (double) targetW / (double) targetH;
        int w = src.getWidth();
        int h = src.getHeight();
        double srcAspect = (double) w / (double) h;
        int cropW = w;
        int cropH = h;
        if (srcAspect > targetAspect) {
            cropW = (int) Math.round(h * targetAspect);
        } else if (srcAspect < targetAspect) {
            cropH = (int) Math.round(w / targetAspect);
        }
        int x = (w - cropW) / 2;
        int y = (h - cropH) / 2;
        BufferedImage cropped = src.getSubimage(x, y, cropW, cropH);
        return resize(cropped, targetW, targetH);
    }

    // Convert to grayscale
    public static BufferedImage toGray(BufferedImage src) {
        BufferedImage gray = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = gray.getGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return gray;
    }

    // Compute variance of Laplacian (focus measure)
    public static double varianceOfLaplacian(BufferedImage src) {
        BufferedImage gray = toGray(src);
        int w = gray.getWidth();
        int h = gray.getHeight();
        int[] pixels = new int[w * h];
        gray.getRaster().getPixels(0, 0, w, h, pixels);
        // 3x3 Laplacian kernel
        int[] dx = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        double[] lap = new double[w * h];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                double acc = 0;
                int idx = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int xx = x + kx;
                        int yy = y + ky;
                        int pv = pixels[yy * w + xx];
                        acc += dx[idx++] * pv;
                    }
                }
                lap[y * w + x] = acc;
            }
        }
        double mean = 0;
        int count = (w - 2) * (h - 2);
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) mean += lap[y * w + x];
        }
        mean /= Math.max(1, count);
        double var = 0;
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                double d = lap[y * w + x] - mean;
                var += d * d;
            }
        }
        return var / Math.max(1, count);
    }

    // Simple 8x8 average hash (aHash) for deduplication
    public static String averageHash(BufferedImage src) {
        BufferedImage gray = toGray(resize(src, 8, 8));
        int w = gray.getWidth();
        int h = gray.getHeight();
        int[] pixels = new int[w * h];
        gray.getRaster().getPixels(0, 0, w, h, pixels);
        long sum = 0;
        for (int v : pixels) sum += v;
        double avg = sum / (double) (w * h);
        long bits = 0L;
        for (int i = 0; i < pixels.length; i++) {
            bits <<= 1;
            if (pixels[i] >= avg) bits |= 1;
        }
        return Long.toHexString(bits);
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
