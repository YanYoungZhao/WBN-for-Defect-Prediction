package CrossVersion;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

public class TotalExecuteForOrigin {
    private static final String res = " ";
    private static final String train = " ";
    private static final String test = " ";
    private static final String suffix = " ";

    public static void main(String[]  args)throws Exception{
        //CrossVer
       // CrossVer crossVer = new CrossVer();
       // crossVer.train = ConverterUtils.DataSource.read(train);
       // crossVer.test = ConverterUtils.DataSource.read(test);
       // crossVer.res = res;
       // crossVer.run(suffix);

        SMOTECrossData smotecrossdata = new SMOTECrossData();
        smotecrossdata.train = ConverterUtils.DataSource.read(train);
        smotecrossdata.test = ConverterUtils.DataSource.read(test);
        smotecrossdata.res = res;
        smotecrossdata.runsmote(suffix);
       /* //BaggingCV
        BaggingCV baggingCV = new BaggingCV();
        baggingCV.train = ConverterUtils.DataSource.read(train);
        baggingCV.test = ConverterUtils.DataSource.read(test);
        baggingCV.res = res;
        baggingCV.run(suffix);
        //AdaBoostCV
        AdaBoostCV adaBoostCV = new AdaBoostCV();
        adaBoostCV.train = ConverterUtils.DataSource.read(train);
        adaBoostCV.test = ConverterUtils.DataSource.read(test);
        adaBoostCV.res = res;
        adaBoostCV.run(suffix);
        //
        VoteCV voteCV  = new VoteCV();
        voteCV.train = ConverterUtils.DataSource.read(train);
        voteCV.test = ConverterUtils.DataSource.read(test);
        voteCV.res = res;
        voteCV.run(suffix);*/
    }
}
