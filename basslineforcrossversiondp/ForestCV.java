package CrossVersion;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.*;
import weka.classifiers.meta.*;
import weka.classifiers.meta.RotationForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class ForestCV {
    //加载数据
    private Instances train;
    private Instances test;
    StringBuffer sb = new StringBuffer();
    ExecutorService threadPools = Executors.newFixedThreadPool(100);
    AtomicInteger sum = new AtomicInteger(0);
    List<String> cls = new ArrayList<>();
    private static final String res = " ";
    public static void main(String[] args) throws Exception {
        ForestCV forestCV = new ForestCV();
        forestCV.train = ConverterUtils.DataSource.read(" ");
        forestCV.test = ConverterUtils.DataSource.read(" ");
        forestCV.run();
    }

    public void run() throws Exception {
        //设置类别标签
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);
        cls.add("RandomForest");
        cls.add("ForestPA");
        cls.add("CSForest");
        for (int i = 0; i < cls.size(); i++) {
            compute(cls.get(i));
        }
    }

    public void compute(String type) throws Exception {
        threadPools.submit(() -> {
            try {

                Classifier classifier = getClassifierType(type);
                classifier.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(classifier, test);
                String classDetails = eval.toClassDetailsString();
                System.out.println(eval.toClassDetailsString());
                String[] arr = classDetails.split("\n");
                sb.append(type+":   "+arr[arr.length-1]).append("\n");
                if(sum.incrementAndGet() == cls.size()){
                    writeData(sb.toString(), res + "ivy-1.4---2.0");
                }
            } catch (Exception e) {
                System.out.println(e);
            }
        });
    }

    public Classifier getClassifierType(String type) {
        switch (type) {
            case "RandomForest":
                return new RandomForest();
              case "ForestPA":
                return new ForestPA();
            case "CSForest":
                return new CSForest();
            default:
                return new RandomForest();
        }
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
            System.out.println("写入" + fileName + "  finish");
        } catch (IOException e) {
            System.err.println(e);
        }
    }
}
