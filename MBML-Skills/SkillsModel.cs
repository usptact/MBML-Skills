using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Utilities;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace MBMLSkills;

static class SkillsModel
{
    public static InferenceResult Run(SkillsDataset data)
    {
        Range person    = new Range(data.NumPersons).Named("persons");
        Range skills    = new Range(data.NumSkills).Named("skills");
        Range questions = new Range(data.NumQuestions).Named("questions");

        var skill = Variable.Array(Variable.Array<bool>(skills), person).Named("skillArray");
        skill[person][skills] = Variable.Bernoulli(0.5).ForEach(person).ForEach(skills);

        var questionSizesArray = Variable.Array<int>(questions).Named("questionSizesArray");
        questionSizesArray.ObservedValue = Util.ArrayInit(data.NumQuestions, q => data.SkillsRequired[q].Length);
        Range questionSizes = new Range(questionSizesArray[questions]).Named("questionSizes");

        var skillsNeeded = Variable.Array(Variable.Array<int>(questionSizes), questions).Named("skillsNeeded");
        skillsNeeded.ObservedValue = data.SkillsRequired;
        skillsNeeded.SetValueRange(skills);

        var probGuess = Variable.Array<double>(questions).Named("probGuess");
        probGuess[questions] = Variable.Beta(2.5, 7.5).ForEach(questions);

        var probNoMistake = Variable.Bernoulli(0.9);

        var relevantSkills = Variable.Array(Variable.Array(Variable.Array<bool>(questionSizes), questions), person);
        var hasSkills      = Variable.Array(Variable.Array<bool>(questions), person);
        var isCorrectVar   = Variable.Array(Variable.Array<bool>(questions), person).Named("isCorrect");

        using (Variable.ForEach(person))
        {
            using (Variable.ForEach(questions))
            {
                relevantSkills[person][questions] = Variable.Subarray<bool>(skill[person], skillsNeeded[questions]);
                hasSkills[person][questions]      = Variable.AllTrue(relevantSkills[person][questions]);

                using (Variable.If(hasSkills[person][questions]))
                    isCorrectVar[person][questions] = probNoMistake;
                using (Variable.IfNot(hasSkills[person][questions]))
                    isCorrectVar[person][questions] = Variable.Bernoulli(probGuess[questions]);
            }
        }

        isCorrectVar.ObservedValue = data.IsCorrect;

        var engine = new InferenceEngine();
        return new InferenceResult(
            engine.Infer<Bernoulli[][]>(skill),
            engine.Infer<Beta[]>(probGuess)
        );
    }
}
