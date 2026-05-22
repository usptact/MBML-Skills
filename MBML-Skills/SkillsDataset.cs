using System;
using System.Linq;

namespace MBMLSkills;

record SkillsDataset(
    int[][]   SkillsRequired,  // [question][skill_index]
    bool[][]  IsCorrect,       // [person][question]
    bool[][]? PersonSkills     // [person][skill] — ground truth for display only; may be null
)
{
    public int NumPersons   => IsCorrect.Length;
    public int NumQuestions => SkillsRequired.Length;

    // When PersonSkills is absent, derive the minimum skill count from SkillsRequired.
    public int NumSkills => PersonSkills is { Length: > 0 }
        ? PersonSkills[0].Length
        : SkillsRequired.SelectMany(q => q).DefaultIfEmpty(-1).Max() + 1;

    public static SkillsDataset Create(int[][] skillsRequired, bool[][] isCorrect, bool[][]? personSkills)
    {
        if (personSkills is not null && personSkills.Length != isCorrect.Length)
            throw new InvalidOperationException(
                $"PersonSkills has {personSkills.Length} persons but IsCorrect has {isCorrect.Length}.");

        int badRow = Array.FindIndex(isCorrect, row => row.Length != skillsRequired.Length);
        if (badRow >= 0)
            throw new InvalidOperationException(
                $"IsCorrect[{badRow}] has {isCorrect[badRow].Length} entries " +
                $"but SkillsRequired has {skillsRequired.Length} questions.");

        if (personSkills is { Length: > 0 })
        {
            int numSkills     = personSkills[0].Length;
            int maxSkillIndex = skillsRequired.SelectMany(q => q).DefaultIfEmpty(-1).Max();
            if (maxSkillIndex >= numSkills)
                throw new InvalidOperationException(
                    $"SkillsRequired references skill index {maxSkillIndex} " +
                    $"but PersonSkills has only {numSkills} columns.");
        }

        return new SkillsDataset(skillsRequired, isCorrect, personSkills);
    }
}
