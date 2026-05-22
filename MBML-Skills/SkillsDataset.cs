using System;
using System.Linq;

namespace MBMLSkills;

record SkillsDataset(
    int[][]  SkillsRequired,  // [question][skill_index]
    bool[][] IsCorrect,       // [person][question]
    bool[][] PersonSkills     // [person][skill] — ground truth for display only
)
{
    public int NumPersons   => IsCorrect.Length;
    public int NumQuestions => SkillsRequired.Length;
    public int NumSkills    => PersonSkills[0].Length;

    public static SkillsDataset Create(int[][] skillsRequired, bool[][] isCorrect, bool[][] personSkills)
    {
        int numSkills     = personSkills[0].Length;
        int maxSkillIndex = skillsRequired.SelectMany(q => q).DefaultIfEmpty(-1).Max();
        if (maxSkillIndex >= numSkills)
            throw new InvalidOperationException(
                $"SkillsRequired references skill index {maxSkillIndex} " +
                $"but PersonSkills has only {numSkills} columns.");
        return new SkillsDataset(skillsRequired, isCorrect, personSkills);
    }
}
