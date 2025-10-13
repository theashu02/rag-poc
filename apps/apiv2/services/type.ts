export interface ReformulatedQueries {
  hypotheticalAnswer: string;
  subQuestions: string[];
  synonymousQuery: string;
}

export interface SparseVector {
  indices: number[];
  values: number[];
}
