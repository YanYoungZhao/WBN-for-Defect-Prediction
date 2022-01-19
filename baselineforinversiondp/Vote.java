package WithinVersion;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

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

public class Vote {

    ExecutorService threadPools = Executors.newFixedThreadPool(100);
    private static final String commPath = " ";
    private static final String res = " ";

    public static void main(String[] args) throws Exception {
        String fileData = commPath + " ";
        Vote vote = new Vote();
        vote.run(fileData);
    }

    public void run(String fileData) throws Exception {
        Instances data1 = new Instances(new BufferedReader(new FileReader(fileData)));
        List<Instances> data = new ArrayList<Instances>();
        data.add(data1);
        data1.setClassIndex(data1.numAttributes() - 1);
        computeAdaboost(data);
    }
    public void computeAdaboost(List<Instances> data) throws Exception{
        String[] paramOptbayesnet = weka.core.Utils.splitOptions("weka.classifiers.meta.Vote -S 1 -B \"weka.classifiers.trees.J48 -C 0.25 -M 2\" -B \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\" -B \"weka.classifiers.lazy.KStar -B 20 -M a\" -B \"weka.classifiers.functions.SGD -F 0 -L 0.01 -R 1.0E-4 -E 500 -C 0.001 -S 1\" -R MAJ");
        weka.classifiers.meta.Vote Vote = new weka.classifiers.meta.Vote();
        Vote.setOptions(paramOptbayesnet);
        compute(data, Vote, "Vote1");

    }
    public void compute(List<Instances> data, Classifier classifier, String flag) throws Exception {
        threadPools.submit(() -> {
            try {
                StringBuilder sb = new StringBuilder();
                for (int k = 1; k <= 100; k++) {
                    Evaluation eval = new Evaluation(data.get(0));
                    eval.crossValidateModel(classifier, data.get(0), 10, new Random(k));
                    String classDetails = eval.toClassDetailsString();
                    System.out.println(eval.toClassDetailsString());
                    String[] arr = classDetails.split("\n");
                    sb.append(arr[arr.length-1]).append("\n");
                }
                writeData(sb.toString(), res +flag);
            } catch (Exception e) {
                System.out.println(e);
            }
        });
    }

    public void writeData(String data, String fileName)
    {
        Path path = Paths.get(fileName + ".txt");
        try {
            if (Files.notExists(path)) {
                Files.createFile(path);
            }
        }catch (IOException e) {
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


