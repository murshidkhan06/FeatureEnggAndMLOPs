// src/featureLabels.ts -- human-friendly names + units for the 10 model inputs.
// The keys are the ML API's field names (see docs/ML_API_CONTRACT.md).
export const FEATURE_LABELS: Record<string, { label: string; unit?: string }> = {
  study_hours: { label: "Study hours", unit: "per week" },
  attendance_pct: { label: "Attendance", unit: "%" },
  mock_test_1: { label: "Mock test 1", unit: "/ 100" },
  mock_test_2: { label: "Mock test 2", unit: "/ 100" },
  mock_test_3: { label: "Mock test 3", unit: "/ 100" },
  income_bracket: { label: "Income bracket" },
  city: { label: "City" },
  enrollment_date: { label: "Enrollment date" },
  shoe_size: { label: "Shoe size", unit: "(not used by the model)" },
  lucky_number: { label: "Lucky number", unit: "(not used by the model)" },
};

export function labelFor(field: string): string {
  return FEATURE_LABELS[field]?.label ?? field;
}

/** Show the score exactly as the API returned it; only force one decimal for whole numbers (95 -> 95.0). */
export function formatScore(value: number): string {
  return Number.isInteger(value) ? value.toFixed(1) : String(value);
}
