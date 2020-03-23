package moa.streams.filters;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
 
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.IntStream;
import moa.core.FastVector; 
import moa.streams.InstanceStream;

/**
 * Random projection filter
 * -d the output dimension (for the dimensionality reduction application)
 *
 * @author Maroua Bahri
 */

public class CS_Filter extends AbstractStreamFilter {


	private static final long serialVersionUID = 1L;
        public IntOption dim = new IntOption("FeatureDimension", 'd', "the target feature dimension.", 10);
        protected InstancesHeader streamHeader;
        protected FastVector attributes;
        protected   double[][] GaussMatrix ;
        Scanner input ;
 
        
	 

	@Override
	public String getPurposeString() {
		return "Creates a random projection.";
	}

	@Override
	public InstancesHeader getHeader() {
 
              return this.streamHeader;
	}

	public Instance filterInstance(Instance x) {


		if(streamHeader==null){
			initialize(x);
		}		
 
		Instance z = transformedInstance(x, GaussianProjection(x,this.dim.getValue(), this.GaussMatrix));
		z.setDataset(streamHeader);

		return z;
	}
        
        
        
	public DenseInstance transformedInstance(Instance sparseInst, double [] val) {
        
        Instances header = this.streamHeader;
        double[] attributeValues = new double[header.numAttributes()];

//        for(int i = 0 ; i < header.numAttributes()-1 ; i++) {
//            attributeValues[i] = val[i];
//        }
        System.arraycopy(val, 0, attributeValues, 0, header.numAttributes()-1);

        attributeValues[attributeValues.length-1] = sparseInst.classValue();
        DenseInstance newInstance = new DenseInstance(1.0, attributeValues);
        newInstance.setDataset(header);
        return newInstance;
    }
        
        public static double[] multiply(double[][] matrix, double[] vector) {
            return Arrays.stream(matrix)
                .mapToDouble(row -> 
                         IntStream.range(0, row.length)
                             .mapToDouble(col -> row[col] * vector[col])
                             .sum()
                    ).toArray();
        }
        
        public  double[] GaussianProjection(Instance instance, int n, double[][] gm) {

        double [] denseValues = new double[n];
      
        double[] ins = new double [instance.numAttributes()-1]; 
        for(int i = 0 ; i < instance.numAttributes()-1 ; i++) {
            ins[i] = instance.value(i);
        }
         denseValues = multiply(gm, ins);
         
        return denseValues;
    }

 

	@Override
	protected void restartImpl() {
		 this.streamHeader = null;
	}

	private void initialize(Instance instance) {
            // System.out.println("init");
             Random r = new Random(System.currentTimeMillis()); 
        this.streamHeader = null;
        this.GaussMatrix = new double[this.dim.getValue()][instance.numAttributes()-1] ;
        for(int i = 0 ; i < this.dim.getValue() ; i++){
            for(int j = 0; j < instance.numAttributes()-1 ; j++){
                this.GaussMatrix[i][j]= r.nextGaussian();
            }
        }
 
    // initialize instance space
		Instances ds = new Instances();
		 
                if (this.streamHeader == null) { 
            //Create a new header
            this.attributes = new FastVector();
            for (int i = 0; i < this.dim.getValue(); i++) {
                this.attributes.addElement(new Attribute("numeric" + (i + 1)));
            } 
            this.attributes.addElement(instance.classAttribute());
            this.streamHeader = new InstancesHeader(new Instances(
                    getCLICreationString(InstanceStream.class), this.attributes, 0));
            this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
              
        }
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
	}

}
