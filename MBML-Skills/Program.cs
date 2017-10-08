//
// Implementation of the "Assessing People's Skills" example from "Model-based Machine Learning"
// Vladislavs Dovgalecs
//

using System;
using System.Collections.Generic;
using System.IO;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;

namespace MBMLSkills
{
    class MainClass
    {
        public static void Main(string[] args)
        {
			//
			// Read data files
			//

			List<string[]> skillsNeededData = GetSkillsNeededData();
            int[][] skillsNeeded = List2Array(skillsNeededData);
            int[] sizes = GetQuestionSizes(skillsNeeded);

            return;

            //
            // ranges
            //

            Range skills = new Range(7);
            Range questions = new Range(sizes.Length);

            //
            // model variables
            //

            VariableArray<bool> skill = Variable.Array<bool>(skills);
            skill[skills] = Variable.Bernoulli(0.5).ForEach(skills);

            // building a jagged 1-D array of 1-D arrays
            VariableArray<int> sizesVar = Variable.Constant(sizes, questions);
            Range feature = new Range(sizesVar[questions]);
            var relevantSkills = Variable.Array(Variable.Array<bool>(feature), questions);

            //
            // model
            //

            using (var b = Variable.ForEach(questions))
            {
                var idx = b.Index;
                //relevantSkills[questions] = Variable.Subarray<bool>(skill, skillsNeeded[idx]);
            }

            //
            // inference
            //
            
            InferenceEngine engine = new InferenceEngine();

            Console.WriteLine("\nPress any key ...");
            Console.ReadKey();
        }

        /// <summary>
        /// Reads "skillsNeeded" matrix from the fixed text file.
        /// </summary>
        /// <returns>The "relevantSkills" list of string[]</returns>
        public static List<string[]> GetSkillsNeededData()
        {
            string path = @"/Users/vlad/Projects/MBML-Skills/MBML-Skills/data/LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv";
            List<string[]> list = new List<string[]>();
            using(var reader = new StreamReader(path))
            {
                while(!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    list.Add(values);
                }
            }
            return list;
        }

        /// <summary>
        /// Converts "skillsNeeded" list repr. to ragged array
        /// </summary>
        /// <returns>The array.</returns>
        /// <param name="list">List.</param>
        public static int[][] List2Array(List<string[]> list)
        {
            int numQuestions = list.Count;
            int[][] skillsNeeded = new int[numQuestions][];
            for (int i = 0; i < numQuestions; i++)
            {
                string[] values = list[i];
                int numSkills = 0;
                for (int j = 0; j < values.Length; j++)
                {
                    if (values[j] == "True")
                        numSkills++;
                }
                skillsNeeded[i] = new int[numSkills];
                int pos = 0;
                for (int j = 0; j < values.Length; j++)
                {
                    if (values[j] == "True") 
                    {
                        skillsNeeded[i][pos] = j;
                        pos++;
                    }
                }
            }
            return skillsNeeded;
        }

        /// <summary>
        /// Gets the question sizes: number of skills per question
        /// </summary>
        /// <returns>The question sizes.</returns>
        /// <param name="list">List.</param>
        public static int[] GetQuestionSizes(int[][] relevantSkills)
        {
            int numQuestions = relevantSkills.Length;
            int[] sizes = new int[numQuestions];
            for (int i = 0; i < numQuestions; i++)
                sizes[i] = relevantSkills[i].Length;
            return sizes;
        }

    }
}
