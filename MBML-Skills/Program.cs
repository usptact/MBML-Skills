//
// Implementation of the "Assessing People's Skills" example from "Model-based Machine Learning"
// Vladislavs Dovgalecs
//

using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.VisualBasic.FileIO;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Utils;

namespace MBMLSkills
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            //
            // Path to data files
            //

            string skillsQuestionsFile = @"LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv";
            skillsQuestionsFile = Path.Combine("..", "..", "data", skillsQuestionsFile);

            string rawResponsesFile = @"LearningSkills_Real_Data_Experiments-Original-Inputs-RawResponsesAsDictionary.csv";
            rawResponsesFile = Path.Combine("..", "..", "data", rawResponsesFile);
            
            //
            // Read data files
            //

            var skillsQuestionsData = ReadCSV(skillsQuestionsFile);
            var rawResponsesData = ReadCSV(rawResponsesFile);

            //
            // Extract data from read files
            //

            int[] trueAnswers = GetTrueAnswersData(rawResponsesData);
            int[][] personAnswers = GetPersonAnswerData(rawResponsesData);
            bool[][] personSkills = GetPersonSkillData(rawResponsesData);       // personSkills[person][skills]

            int[][] skillsRequired = GetSkillsNeededData(skillsQuestionsData);  // skillsRequired[question][skill]

            int numPersons = personAnswers.Length;
            int numSkills = personSkills[0].Length;
            int numQuestions = skillsRequired.Length;

            //
            // ranges
            //

            Range person = new Range(numPersons).Named("persons");
            Range skills = new Range(numSkills).Named("skills");
            Range questions = new Range(numQuestions).Named("questions");

            //
            // model variables
            //

            VariableArray<bool> skill = Variable.Array<bool>(skills).Named("skillArray");
            skill[skills] = Variable.Bernoulli(0.5).ForEach(skills);

            // helper variable array: question sizes
            // each question has some number of skills to be answered correctly
            var questionSizesArray = Variable.Array<int>(questions).Named("questionSizesArray");
            questionSizesArray.ObservedValue = Util.ArrayInit(numQuestions, q => skillsRequired[q].Length);
            Range questionSizes = new Range(questionSizesArray[questions]).Named("questionSizes");

            // skillsNeeded: building a jagged 1-D array of 1-D arrays
            var skillsNeeded = Variable.Array(Variable.Array<int>(questionSizes), questions).Named("skillsNeeded");
            skillsNeeded.ObservedValue = skillsRequired;
            skillsNeeded.SetValueRange(skills);

            var relevantSkills = Variable.Array(Variable.Array<bool>(questionSizes), questions);

            VariableArray<bool> hasSkills = Variable.Array<bool>(questions);
            VariableArray<bool> isCorrect = Variable.Array<bool>(questions);

            //
            // model
            //

            using (Variable.ForEach(questions))
            {
                // pick subset of skills that are needed to answer the question
                relevantSkills[questions] = Variable.Subarray<bool>(skill, skillsNeeded[questions]);

                // all skills are required to answer the question
                hasSkills[questions] = Variable.AllTrue(relevantSkills[questions]);

                // AddNoise logic
                using (Variable.If(hasSkills[questions]))
                    isCorrect[questions] = Variable.Bernoulli(0.9);
                using (Variable.IfNot(hasSkills[questions]))
                    isCorrect[questions] = Variable.Bernoulli(0.1);
            }

            //
            // inference
            //

            InferenceEngine engine = new InferenceEngine();

            for (int i = 0; i < numPersons; i++)
            {
                isCorrect.ObservedValue = BuildIsCorrect(trueAnswers, personAnswers[i]);
                Bernoulli[] hasSkillsMarginal = engine.Infer<Bernoulli[]>(hasSkills);
                Console.WriteLine("PERSON #{0} has skills: ", i+1);
                for (int j = 0; j < numSkills; j++){
                    string s = string.Format("{0:N3}", hasSkillsMarginal[j].GetProbTrue());
                    Console.Write(s);
                    Console.Write(" ");
                }
                Console.WriteLine("");
            }

            Console.WriteLine("\nPress any key ...");
            Console.ReadKey();
        }

        public static bool[] BuildIsCorrect(int[] trueAnswers, int[] personAnswers)
        {
            int numQuestions = trueAnswers.Length;
            bool[] isCorrect = new bool[numQuestions];
            for (int i = 0; i < numQuestions; i++)
                if (trueAnswers[i] == personAnswers[i])
                    isCorrect[i] = true;
                else
                    isCorrect[i] = false;
            return isCorrect;
        }

        /// <summary>
        /// Reads the csv file into a list of fields (string[])
        /// </summary>
        /// <returns>List of fields</returns>
        /// <param name="path">Full path to a CSV file</param>
        public static List<string[]> ReadCSV(string path)
        {
            List<string[]> list = new List<string[]>();
            using (TextFieldParser parser = new TextFieldParser(@path))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");
                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    list.Add(fields);
                }
            }
            return list;
        }

        public static int[] GetTrueAnswersData(List<string[]> list)
        {
            string[] fields = list[1];
            int numQuestions = fields.Length - 8; // minus 1 id and 7 skill columns
            int[] trueAnswers = new int[numQuestions];
            for (int i = 8; i < fields.Length; i++)
                trueAnswers[i - 8] = Int32.Parse(fields[i]);
            return trueAnswers;
        }

        public static int[][] GetPersonAnswerData(List<string[]> list)
        {
            int numPersons = list.Count - 2;    // minus header and true answers lines
            int[][] personAnswers = new int[numPersons][];
            for (int i = 2; i < list.Count; i++)
            {
                string[] fields = list[i];
                int numQuestions = fields.Length - 8;
                personAnswers[i - 2] = new int[numQuestions];
                for (int j = 8; j < fields.Length; j++)
                    personAnswers[i - 2][j - 8] = Int32.Parse(fields[j]);
            }
            return personAnswers;
        }

        public static bool[][] GetPersonSkillData(List<string[]> list)
        {
            int numPersons = list.Count - 2;
            bool[][] personSkills = new bool[numPersons][];
            for (int i = 2; i < list.Count; i++)
            {
                string[] fields = list[i];
                int numSkills = 7;
                personSkills[i - 2] = new bool[numSkills];
                for (int j = 1; j < 8; j++)
                {
                    if (fields[j] == "True")
                        personSkills[i - 2][j - 1] = true;
                    if (fields[j] == "False")
                        personSkills[i - 2][j - 1] = false;
				}
            }
            return personSkills;
        }

        /// <summary>
        /// Returns skillsNeeded data from a list.
        /// See ReadCSV() to read the CSV file first into a list.
        /// </summary>
        /// <returns>The array.</returns>
        /// <param name="list">List.</param>
        public static int[][] GetSkillsNeededData(List<string[]> list)
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

    }
}
