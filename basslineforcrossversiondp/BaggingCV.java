package CrossVersion;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.classifiers.meta.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class BaggingCV {

    //加载数据
    public Instances train;
    public Instances test;
    StringBuffer sb = new StringBuffer();
    ExecutorService threadPools = Executors.newFixedThreadPool(100);
    AtomicInteger sum = new AtomicInteger(0);
    List<String> cls = new ArrayList<>();
    Integer size = 2;
    public String res;

    public static void main(String[] args) throws Exception {

    }

    public void run(String suffix) throws Exception {
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

 

        String[] paramOptJ48 = weka.core.Utils.splitOptions("-P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2");
        Bagging baggingJ48 = new Bagging();
        baggingJ48.setOptions(paramOptJ48);
        compute(baggingJ48, "J48",suffix);

    }
    public void compute(Classifier classifier, String flag,String suffix) throws Exception {
        threadPools.submit(() -> {
            try {
                classifier.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(classifier, test);
                String classDetails = eval.toClassDetailsString();
                System.out.println(eval.toClassDetailsString());
                String[] arr = classDetails.split("\n");
                sb.append(flag + ":   " + arr[arr.length - 1] + "\n");
                if (sum.incrementAndGet() == size) {
                    writeData(sb.toString(), res + "bagging"+suffix);
                }
            } catch (Exception e) {
                System.out.println(e);
            }
        });
    }

    public void writeData(String data, String fileName) {
        Path path = Paths.get(fileName + ".txt");
        try {
            if (Files.notExists(path)) {
                Files.createFile(path);
            }
        } catch (IOException e) {
            System.err.println(e);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write(data);
            System.out.println("写入" + fileName + " finish");
        } catch (IOException e) {
            System.err.println(e);
        }
    }
}

