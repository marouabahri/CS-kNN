/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.lazy;

import java.io.StringReader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;
import moa.core.FastVector;
import moa.streams.InstanceStream;

/**
 * k Nearest Neighbor.<p>
 *
 * -k number of neighbours <br> -m max instances <br>
 * -w the size of the window (the number of instances to store inside the window)
 * -d the output dimension (for the dimensionality reduction application)
 * -f the number of features in the input space (e.g. a dataset with 100 features)
 *
 * @author Maroua Bahri
 */
public class CS_kNN extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

	public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

	public IntOption limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

    public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
                "LinearNN", "KDTree"},
            new String[]{"Brute force search algorithm for nearest neighbour search. ",
                "KDTree search algorithm for nearest neighbour search"
            }, 0);
    public IntOption dim = new IntOption("OutputFeatureDimension", 'd',
            "the target feature dimension.", 10);
          //The number of features in the input dataset (put the number of features only, do not count the class label as a feature)
    public IntOption NumAttributes = new IntOption("InputFeatureDimension", 'f', "the input feature dimension.", 1000);

    protected InstancesHeader streamHeader;
	int C = 0;
	protected   double[][] GaussMatrix ;
    protected FastVector attributes;
    Scanner input ;

    @Override
    public String getPurposeString() {
        return "kNN: special.";
    }

    protected Instances window; 

	@Override
	public void setModelContext(InstancesHeader context) {
                 if (streamHeader == null) {
            //Create a new header
            this.attributes = new FastVector();
            for (int i = 0; i < this.dim.getValue(); i++) {
                this.attributes.addElement(new Attribute("numeric" + (i + 1)));
            } 
            this.attributes.addElement(context.classAttribute());
            this.streamHeader = new InstancesHeader(new Instances(
                    getCLICreationString(InstanceStream.class), this.attributes, 0));

            this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        }
                context = this.streamHeader;
            try {
			this.window = new Instances(context,0); //new StringReader(context.toString())
			this.window.setClassIndex(context.classIndex());
		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			//e.printStackTrace();
			//System.exit(1);
		}
 
	}

    @Override
    public void resetLearningImpl() {
		this.window = null;
		Random r = new Random();
        this.streamHeader = null; 
        
        this.GaussMatrix = new double[this.dim.getValue()][this.NumAttributes.getValue()] ;
        for(int i = 0 ; i < this.dim.getValue() ; i++){
            for(int j = 0; j < this.NumAttributes.getValue() ; j++){
                this.GaussMatrix[i][j]= r.nextGaussian();
            }
        }

//        try {
//            input = new Scanner(new BufferedReader(new
 //       FileReader(String.format("/home/mbahri/Bureau/tsneUmap/datasets/gaussianMatrices/mat3.txt"))));
 //       } catch (FileNotFoundException ex) {
//            System.out.println("exception gaussian matrix");
//        }

//        this.GaussMatrix = new double[this.dim.getValue()][this.NumAttributes.getValue()] ;
//        for(int i = 0 ; i < this.dim.getValue() ; i++){
//            String [] line = input.nextLine().trim().split(",");
//            for(int j = 0; j < this.NumAttributes.getValue() ; j++){
//                this.GaussMatrix[i][j] = Double.parseDouble(line[j]);
//
//            }
//        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
         
	    if (streamHeader == null) {
            //Create a new header
            this.attributes = new FastVector();
            for (int i = 0; i < this.dim.getValue(); i++) {
                this.attributes.addElement(new Attribute("numeric" + (i + 1)));
            }
            this.attributes.addElement(inst.classAttribute());
            this.streamHeader = new InstancesHeader(new Instances(
                    getCLICreationString(InstanceStream.class), this.attributes, 0));
            this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        }
 
        inst = transformedInstance(inst, 
                GaussianProjection(inst,this.dim.getValue(), GaussMatrix));
        
		if (inst.classValue() > C)
		C = (int)inst.classValue(); 
 
                if (this.window == null) { 
			this.window = new Instances(inst.dataset());
		}
		if (this.limitOption.getValue() <= this.window.numInstances()) {
			this.window.delete(0);
		}
  
		this.window.add(inst); 
    }
    
    public DenseInstance transformedInstance(Instance sparseInst, double [] val) {
        
        Instances header = this.streamHeader;
        double[] attributeValues = new double[header.numAttributes()];

        System.arraycopy(val, 0, attributeValues, 0, header.numAttributes()-1);

        attributeValues[attributeValues.length-1] = sparseInst.classValue();
        DenseInstance newInstance = new DenseInstance(1.0, attributeValues);
        newInstance.setDataset(header);
        return newInstance;
    }

    public  double[] GaussianProjection(Instance instance, int n, double[][] gm) {
       

        double [] denseValues;
      
        double[] ins = new double [instance.numAttributes()-1]; 
        for(int i = 0 ; i < instance.numAttributes()-1 ; i++) {
            ins[i] = instance.value(i);
        }
         denseValues = multiply(gm, ins);
         
        return denseValues;
    }

    public static double[] multiply(double[][] a, double[] x) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += a[i][j] * x[j];
        return y;
    }

	@Override
    public double[] getVotesForInstance(Instance inst) {
         if (streamHeader == null) {
            //Create a new header
            this.attributes = new FastVector();
            for (int i = 0; i < this.dim.getValue(); i++) {
                this.attributes.addElement(new Attribute("numeric" + (i + 1)));
            }
            this.attributes.addElement(inst.classAttribute());
            this.streamHeader = new InstancesHeader(new Instances(
                    getCLICreationString(InstanceStream.class), this.attributes, 0));
            this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        }
        inst = transformedInstance(inst, 
                GaussianProjection(inst,this.dim.getValue(), GaussMatrix));
             
		double v[] = new double[C+1];
		try {
			NearestNeighbourSearch search;
			if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
				search = new LinearNNSearch(this.window);  
			} else {
				search = new KDTree();
				search.setInstances(this.window);
			}	
			if (this.window.numInstances()>0) {	
				Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.window.numInstances()));
				for(int i = 0; i < neighbours.numInstances(); i++) {
					v[(int)neighbours.instance(i).classValue()]++;
				}
			}
		} catch(Exception e) {
			//System.err.println("Error: kNN search failed.");
			//e.printStackTrace();
			//System.exit(1);
			return new double[inst.numClasses()];
		}
		return v;
    }

    
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }
}