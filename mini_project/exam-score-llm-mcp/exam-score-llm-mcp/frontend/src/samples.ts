// src/samples.ts -- example QUESTIONS (not results). Clicking one only fills the input box;
// the prediction always comes from the real model.

export interface Sample {
  title: string;
  what: string; // what this sample demonstrates
  text: string;
}

export const SAMPLES: Sample[] = [
  {
    title: "Complete profile",
    what: "Every field given: the model runs.",
    text:
      "Predict the exam score for a student who studies 12.5 hours per week, has 88% attendance, " +
      "scored 72, 75 and 70 in the three mock tests, belongs to the Medium income bracket, lives in Pune, " +
      "enrolled on 2025-06-01, has shoe size 9 and lucky number 42.",
  },
  {
    title: "Missing details",
    what: "The bot should ask, not guess.",
    text: "Predict the exam score for a student who studied 6 hours daily, attended 85% of classes, and completed 8 assignments.",
  },
  {
    title: "Compare two students",
    what: "Two profiles, two model calls.",
    text:
      "Compare two students. Student A: 10 hours per week, 80% attendance, mock tests 60, 65, 62, Low income, Delhi, enrolled 2025-01-15, shoe size 8, lucky number 7. " +
      "Student B: same as A but 15 hours per week.",
  },
  {
    title: "What does the model need?",
    what: "Asks the MCP model-information tool.",
    text: "What inputs does the exam score model need, and what are their allowed ranges?",
  },
  {
    title: "Unrelated question",
    what: "No model call should happen.",
    text: "What is the capital of France?",
  },
];
