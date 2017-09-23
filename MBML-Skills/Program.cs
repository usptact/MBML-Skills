//
// Implementation of the "Assessing People's Skills" example from "Model-based Machine Learning"
// Vladislavs Dovgalecs
//

using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;

namespace MBMLSkills
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            // hidden RVs, fixed factor Bernoulli(0.5)
            Variable<bool> csharp = Variable.Bernoulli(0.5);
            Variable<bool> sql = Variable.Bernoulli(0.5);

            // declare variables but don't define statistically yet (no factor attached)
            Variable<bool> isCorrect1 = Variable.New<bool>();
            Variable<bool> isCorrect2 = Variable.New<bool>();
            Variable<bool> isCorrect3 = Variable.New<bool>();

            // defined as AND(csharp, sql)
            Variable<bool> hasSkills = csharp & sql;

            // set CPT for isCorrect1
            using (Variable.If(csharp))
                isCorrect1.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(csharp))
				isCorrect1.SetTo(Variable.Bernoulli(0.2));

			// set CPT for isCorrect2
			using (Variable.If(sql))
				isCorrect2.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(sql))
				isCorrect2.SetTo(Variable.Bernoulli(0.2));

			// set CPT for isCorrect3
			using (Variable.If(hasSkills))
				isCorrect3.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(hasSkills))
				isCorrect3.SetTo(Variable.Bernoulli(0.2));
            
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;

            // setting observed values and doing inference
            // set different observed values and see the result
            isCorrect1.ObservedValue = true;
            isCorrect2.ObservedValue = false;
            isCorrect3.ObservedValue = true;
            Console.WriteLine("P(csharp=True|isCorrect1=true, isCorrect2=false, isCorrect3=true) = {0}", engine.Infer<Bernoulli>(csharp).GetProbTrue());
            Console.WriteLine("P(sql=True|isCorrect1=true, isCorrect2=false, isCorrect3=true) = {0}", engine.Infer<Bernoulli>(sql).GetProbTrue());

            Console.WriteLine("\nPress any key ...");
            Console.ReadKey();
        }
    }
}
