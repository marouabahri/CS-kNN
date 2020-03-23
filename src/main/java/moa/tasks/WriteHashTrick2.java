/*
 *    WriteStreamToARFFFile.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
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
package moa.tasks;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.Writer;

import moa.core.ObjectRepository;
import moa.options.ClassOption;
import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
//import gov.sandia.cognition.learning.data.feature.FeatureHashing;
//import gov.sandia.cognition.math.matrix.Vector;
//import gov.sandia.cognition.math.matrix.VectorFactory;
import moa.streams.InstanceStream;
//import de.jungblut.nlp.VectorizerUtils;

/**
 * Task to output a stream to an ARFF file
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class WriteHashTrick2 extends AuxiliarMainTask {

    @Override
    public String getPurposeString() {
        return "Outputs a stream to an ARFF file.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to write.", InstanceStream.class,
            "generators.RandomTreeGenerator");

    public FileOption arffFileOption = new FileOption("arffFile", 'f',
            "Destination ARFF file.", null, "arff", true);

    public IntOption maxInstancesOption = new IntOption("maxInstances", 'm',
            "Maximum number of instances to write to file.", 1000000, 0,
            Integer.MAX_VALUE);
    public IntOption nValue = new IntOption("nTargetfeat", 'n',
            "n the target dimension of the vector.", 1);
 
    public FlagOption suppressHeaderOption = new FlagOption("suppressHeader",
            'h', "Suppress header from output.");
   // protected FeatureHashing feathash = new FeatureHashing(this.nValue.getValue());

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
        System.out.println("hi");;
        InstanceStream stream = (InstanceStream) getPreparedClassOption(this.streamOption);
        File destFile = this.arffFileOption.getFile();
        if (destFile != null) { 
            try {
                Writer w = new BufferedWriter(new FileWriter(destFile));
                monitor.setCurrentActivityDescription("Writing stream to ARFF");
                if (!this.suppressHeaderOption.isSet()) {
                    w.write(stream.getHeader().toString());
                    w.write("\n");
                }
                int numWritten = 0;  
                while ((numWritten < this.maxInstancesOption.getValue())
                        && stream.hasMoreInstances()) {
                    System.out.println("while");

                    System.out.println(stream.nextInstance().getData().toString());
                   // Vector vec =createVector(stream.nextInstance().getData().toString());
                   //   w.write(this.feathash.evaluate(vec).toString());
                    w.write("\n");
                    numWritten++;
                }
                w.close();
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Failed writing to file " + destFile, ex);
            }
            return "Stream written to ARFF file " + destFile;
        }
        System.out.println("bye");
        throw new IllegalArgumentException("No destination file to write to.");
    }

    
      /*  private Vector createVector(String row) {
        VectorFactory<?> vf = VectorFactory.getSparseDefault();
        String [] data= row.split(",");
        int size = data.length;
      Vector input = VectorFactory.getDefault().createVector(data.length);
     
      for(int i = 0 ; i< data.length-1; i++){

           input.setElement(i, Double.valueOf(data[i]));
      }
      return input ;
    } */
    @Override
    public Class<?> getTaskResultType() {
        return String.class;
    }
}
