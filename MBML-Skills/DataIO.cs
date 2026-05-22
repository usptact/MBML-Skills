using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.VisualBasic.FileIO;

namespace MBMLSkills;

static class DataIO
{
    // ─── Synthetic format ──────────────────────────────────────────────────────

    public static void Write(string dir, SkillsDataset data)
    {
        Directory.CreateDirectory(dir);
        WriteBoolMatrix(Path.Combine(dir, "skills_required.csv"), IndicesToMask(data.SkillsRequired, data.NumSkills));
        WriteBoolMatrix(Path.Combine(dir, "is_correct.csv"), data.IsCorrect);
        WriteBoolMatrix(Path.Combine(dir, "person_skills.csv"), data.PersonSkills);
    }

    public static SkillsDataset ReadSynthetic(string dir) => new(
        ReadSkillsRequired(Path.Combine(dir, "skills_required.csv")),
        ReadBoolMatrix(Path.Combine(dir, "is_correct.csv")),
        ReadBoolMatrix(Path.Combine(dir, "person_skills.csv"))
    );

    // ─── Real data format ──────────────────────────────────────────────────────

    public static SkillsDataset ReadReal(string dir)
    {
        var skillsData   = ReadCSV(Path.Combine(dir,
            "LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv"));
        var responsesData = ReadCSV(Path.Combine(dir,
            "LearningSkills_Real_Data_Experiments-Original-Inputs-RawResponsesAsDictionary.csv"));

        return new SkillsDataset(
            ParseSkillsRequired(skillsData),
            BuildIsCorrect(ParseTrueAnswers(responsesData), ParsePersonAnswers(responsesData)),
            ParsePersonSkills(responsesData)
        );
    }

    // ─── Synthetic helpers ─────────────────────────────────────────────────────

    static bool[][] IndicesToMask(int[][] indices, int numSkills)
    {
        return indices.Select(row =>
        {
            bool[] mask = new bool[numSkills];
            foreach (int i in row) mask[i] = true;
            return mask;
        }).ToArray();
    }

    static void WriteBoolMatrix(string path, bool[][] matrix)
    {
        using var writer = new StreamWriter(path);
        foreach (var row in matrix)
            writer.WriteLine(string.Join(",", row.Select(b => b ? 1 : 0)));
    }

    static int[][] ReadSkillsRequired(string path) =>
        File.ReadAllLines(path)
            .Select(line => line.Split(',')
                .Select((v, i) => (v, i))
                .Where(x => x.v == "1")
                .Select(x => x.i)
                .ToArray())
            .ToArray();

    static bool[][] ReadBoolMatrix(string path) =>
        File.ReadAllLines(path)
            .Select(line => line.Split(',').Select(v => v == "1").ToArray())
            .ToArray();

    // ─── Real data helpers ─────────────────────────────────────────────────────

    static List<string[]> ReadCSV(string path)
    {
        var list = new List<string[]>();
        using var parser = new TextFieldParser(path);
        parser.TextFieldType = FieldType.Delimited;
        parser.SetDelimiters(",");
        while (!parser.EndOfData)
            list.Add(parser.ReadFields()!);
        return list;
    }

    static int[] ParseTrueAnswers(List<string[]> list)
    {
        string[] fields = list[1];
        return Enumerable.Range(8, fields.Length - 8).Select(i => int.Parse(fields[i])).ToArray();
    }

    static int[][] ParsePersonAnswers(List<string[]> list) =>
        list.Skip(2).Select(fields =>
            Enumerable.Range(8, fields.Length - 8).Select(i => int.Parse(fields[i])).ToArray()
        ).ToArray();

    static bool[][] ParsePersonSkills(List<string[]> list) =>
        list.Skip(2).Select(fields =>
            Enumerable.Range(1, 7).Select(j => fields[j] == "True").ToArray()
        ).ToArray();

    static int[][] ParseSkillsRequired(List<string[]> list) =>
        list.Select(values =>
            values.Select((v, i) => (v, i)).Where(x => x.v == "True").Select(x => x.i).ToArray()
        ).ToArray();

    static bool[][] BuildIsCorrect(int[] trueAnswers, int[][] personAnswers) =>
        personAnswers.Select(answers =>
            Enumerable.Range(0, trueAnswers.Length).Select(q => trueAnswers[q] == answers[q]).ToArray()
        ).ToArray();
}
