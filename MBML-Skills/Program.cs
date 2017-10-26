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
            /* Uncomment for synthetic data
            int numSkills = 5;
            int numQuestions = 10;
            int numPersons = 50;

            SampleSkillsNeededData(numSkills, numQuestions, out int[][] skillsNeededData);
            SamplePersonSkillsData(numSkills, numPersons, out bool[][] personSkillsData);
            SampleIsCorrectData(numPersons, numQuestions, numSkills, skillsNeededData, personSkillsData, out bool[][] isCorrectData);
            */
		
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
                isCorrect[questions] = AddNoise(hasSkills[questions]);
            }

            //
            // inference
            //

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;

            for (int p = 0; p < numPersons; p++)
            {
                //var pSkills = personSkillsData[p];
                //isCorrect.ObservedValue = isCorrectData[p];

                isCorrect.ObservedValue = BuildIsCorrect(trueAnswers, personAnswers[p]);

                Bernoulli[] skillMarginal = engine.Infer<Bernoulli[]>(skill);

                // print person skill marginals
                Console.WriteLine("PERSON #{0} has skills: ", p+1);
                for (int j = 0; j < numSkills; j++){
                    string s = string.Format("{0:N3}", skillMarginal[j].GetProbTrue());
                    Console.Write(s);
                    Console.Write(" ");
                }

                // print groundtruth skill indicators
                Console.WriteLine("");
                for (int j = 0; j < numSkills; j++)
                {
                    string s = string.Format("{0:N3}", personSkills[p][j]);
                    Console.Write(s);
                    Console.Write(" ");
                }
                Console.WriteLine("");
                Console.WriteLine("");
            }

            Console.WriteLine("\nPress any key ...");
            Console.ReadKey();
        }
	    
        /// <summary>
        /// AddNoise factor
        /// </summary>
        /// <param name="hasSkills">True hasSkills RV</param>
        /// <returns>Noisy version of the hasSkills RV</returns>
        public static Variable<bool> AddNoise(Variable<bool> hasSkills)
        {
            var noisyIsCorrect = Variable.New<bool>();
            using (Variable.If(hasSkills))
                noisyIsCorrect.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(hasSkills))
                noisyIsCorrect.SetTo(Variable.Bernoulli(0.1));
            return noisyIsCorrect;
        }

        /// <summary>
        /// Synthetic data: Samples question skill data
        /// </summary>
        /// <param name="numSkills">total number of skills</param>
        /// <param name="numQuestions">total number of questions</param>
        /// <param name="skillsNeeded">output array of arrays</param>
	    public static void SampleSkillsNeededData(int numSkills, int numQuestions, out int[][] skillsNeeded)
        {
            skillsNeeded = new int[numQuestions][];
            var coin = new Bernoulli(0.5);
            for (int q = 0; q < numQuestions; q++)
            {
                List<int> skillsList = new List<int>();
                for (int s = 0; s < numSkills; s++)
                    if (coin.Sample())
                        skillsList.Add(s);
                int L = skillsList.Count;
                skillsNeeded[q] = new int[L];
                for (int s = 0; s < L; s++)
                    skillsNeeded[q][s] = skillsList[s];
            }
        }

        /// <summary>
        /// Synthetic data: Samples person skill data
        /// </summary>
        /// <param name="numSkills">total number of skills</param>
        /// <param name="numPersons">total number of persons</param>
        /// <param name="personSkills">output array of arrays</param>
        public static void SamplePersonSkillsData(int numSkills, int numPersons, out bool[][] personSkills)
        {
            personSkills = new bool[numPersons][];
            var coin = new Bernoulli(0.5);
            for (int p = 0; p < numPersons; p++)
            {
                personSkills[p] = new bool[numSkills];
                for (int s = 0; s < numSkills; s++)
                    personSkills[p][s] = coin.Sample();
            }
        }

        /// <summary>
        /// Synthetic data: Samples answer data
        /// </summary>
        /// <param name="numPersons">total number of persons</param>
        /// <param name="numQuestions">total number of questions</param>
        /// <param name="numSkills">total number of skills</param>
        /// <param name="skillsNeeded">question skills array of arrays</param>
        /// <param name="personSkills">person skills array of arrays</param>
        /// <param name="isCorrect">output array of arrays</param>
        public static void SampleIsCorrectData(int numPersons, int numQuestions, int numSkills,
                                               int[][] skillsNeeded, bool[][] personSkills,
                                               out bool[][] isCorrect)
        {
            Bernoulli isCorrectCoin = new Bernoulli(0.9);
            Bernoulli isIncorrectCoin = new Bernoulli(0.1);
            isCorrect = new bool[numPersons][];
            for (int p = 0; p < numPersons; p++)
            {
                var pSkills = personSkills[p];              // skills of p-th person
                isCorrect[p] = new bool[numQuestions];
                for (int q = 0; q < numQuestions; q++)
                {
                    var qSkills = skillsNeeded[q];          // skills needed to answer q-th question
                    bool hasSkills = false;
                    for (int s = 0; s < qSkills.Length; s++)
                    {
                        if (s == 0)
                            hasSkills = pSkills[qSkills[s]];
                        else
                            hasSkills = hasSkills & pSkills[qSkills[s]];
                    }
                    if (hasSkills)
                        isCorrect[p][q] = isCorrectCoin.Sample();
                    else
                        isCorrect[p][q] = isIncorrectCoin.Sample();
                }
            }
        }

        /// <summary>
        /// Build isCorrect output comparing person answers to groundtruth
        /// </summary>
        /// <param name="trueAnswers">array of groundtruth answers</param>
        /// <param name="personAnswers">array of person answers</param>
        /// <returns>array of indicators of whether each question was correctly answered</returns>
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

        /// <summary>
        /// Parses groundtruth question answers
        /// </summary>
        /// <param name="list">data structure</param>
        /// <returns>array of integers</returns>
        public static int[] GetTrueAnswersData(List<string[]> list)
        {
            string[] fields = list[1];
            int numQuestions = fields.Length - 8; // minus 1 id and 7 skill columns
            int[] trueAnswers = new int[numQuestions];
            for (int i = 8; i < fields.Length; i++)
                trueAnswers[i - 8] = Int32.Parse(fields[i]);
            return trueAnswers;
        }

        /// <summary>
        /// Parses person answer data
        /// </summary>
        /// <param name="list">data structure</param>
        /// <returns>array of array of ints</returns>
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

        /// <summary>
        /// Parses person skill self-assessment data
        /// </summary>
        /// <param name="list">data structure</param>
        /// <returns>array of array of bool</returns>
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
