package org.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import org.mapreduce.frugalnerf.FrugalNeRFMapper;
import org.mapreduce.frugalnerf.FrugalNeRFReducer;
import org.mapreduce.utils.ImageInputFormat;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: PreprocessJob <input_hdfs_dir> <output_hdfs_dir>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Image Preprocess Job");

        job.setJarByClass(Main.class);

        // Input format (custom), mapper, reducer
        job.setInputFormatClass(ImageInputFormat.class);
        job.setMapperClass(FrugalNeRFMapper.class);
        job.setReducerClass(FrugalNeRFReducer.class);

        // Map output
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(BytesWritable.class);

        // Final output (SequenceFile of Text -> BytesWritable)
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(BytesWritable.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        ImageInputFormat.addInputPath(job, new Path(args[0]));
        SequenceFileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}