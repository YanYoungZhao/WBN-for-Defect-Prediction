package CrossVersion;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

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

public class VoteCV {

    //加载数据
    public Instances train;
    public Instances test;
    StringBuffer sb = new StringBuffer();
    ExecutorService threadPools = Executors.newFixedThreadPool(100);
    AtomicInteger sum = new AtomicInteger(0);
    List<String> cls = new ArrayList<>();
    Integer size = 1;
    public String res;

    public static void main(String[] args) throws Exception {

    }
    public void run(String suffix) throws Exception {
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);
        String[] paramOptbayesnet = weka.core.Utils.splitOptions("weka.classifiers.meta.Vote -S 1 -B \"weka.classifiers.trees.REPTree -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0\" -B \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\" -B \"weka.classifiers.bayes.NaiveBayes \" -B \"weka.classifiers.functions.SGD -F 0 -L 0.01 -R 1.0E-4 -E 500 -C 0.001 -S 1\" -R MAJ");
        weka.classifiers.meta.Vote vote = new weka.classifiers.meta.Vote();
        vote.setOptions(paramOptbayesnet);
        compute(vote, "Vote1",suffix);
    }

    public void compute(Classifier classifier, String flag,String suffix) throws Exception {
        threadPools.submit(() -> {
            try {
                classifier.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(classifier, test);
                String classDetails = eval.toClassDetailsString();
                String[] arr = classDetails.split("\n");
                sb.append(flag + ":   " + arr[arr.length - 1] + "\n");
                if (sum.incrementAndGet() == size) {
                    writeData(sb.toString(), res + "vote"+suffix);}
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

