from pydantic import BaseModel, Field
from typing import Optional

# 定义选项与数值的映射
GENDER_MAPPING = {"Male": 1, "Female": 0}
BINARY_MAPPING = {"Yes": 1, "No": 0}


class UserInput(BaseModel):
    # 数值字段
    Pregnancies: Optional[int] = Field(None, description="Number of times pregnant")
    Glucose: Optional[float] = Field(None, description="Plasma glucose concentration")
    BloodPressure: Optional[float] = Field(None, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: Optional[float] = Field(None, description="Triceps skin fold thickness (mm)")
    Insulin: Optional[float] = Field(None, description="2-Hour serum insulin (mu U/ml)")
    BMI: Optional[float] = Field(None, description="Body mass index")
    DiabetesPedigreeFunction: Optional[float] = Field(None, description="Diabetes pedigree function")
    Age: Optional[int] = Field(None, description="Age in years")
    Urea: Optional[float] = Field(None, description="Urea level")
    Cr: Optional[float] = Field(None, description="Serum creatinine level")
    HbA1c: Optional[float] = Field(None, description="Hemoglobin A1c level")
    Chol: Optional[float] = Field(None, description="Cholesterol level")
    TG: Optional[float] = Field(None, description="Triglycerides level")
    HDL: Optional[float] = Field(None, description="High-density lipoprotein level")
    LDL: Optional[float] = Field(None, description="Low-density lipoprotein level")
    VLDL: Optional[float] = Field(None, description="Very low-density lipoprotein level")

    # 二分类字段
    Gender: Optional[str] = Field(None, description="Gender (Male/Female)")
    Polyuria: Optional[str] = Field(None, description="Frequent urination (Yes/No)")
    Polydipsia: Optional[str] = Field(None, description="Excessive thirst (Yes/No)")
    SuddenWeightLoss: Optional[str] = Field(None, description="Sudden weight loss (Yes/No)")
    Weakness: Optional[str] = Field(None, description="Feeling weak (Yes/No)")
    Polyphagia: Optional[str] = Field(None, description="Excessive hunger (Yes/No)")
    GenitalThrush: Optional[str] = Field(None, description="Genital thrush (Yes/No)")
    VisualBlurring: Optional[str] = Field(None, description="Blurred vision (Yes/No)")
    Itching: Optional[str] = Field(None, description="Persistent itching (Yes/No)")
    Irritability: Optional[str] = Field(None, description="Feeling irritable (Yes/No)")
    DelayedHealing: Optional[str] = Field(None, description="Delayed wound healing (Yes/No)")
    PartialParesis: Optional[str] = Field(None, description="Partial muscle paralysis (Yes/No)")
    MuscleStiffness: Optional[str] = Field(None, description="Muscle stiffness (Yes/No)")
    Alopecia: Optional[str] = Field(None, description="Hair loss (Yes/No)")
    Obesity: Optional[str] = Field(None, description="Obesity (Yes/No)")

    def to_model_input(self):
        """将用户输入转换为模型可以处理的数值格式"""
        return [
            self.Pregnancies,
            self.Glucose,
            self.BloodPressure,
            self.SkinThickness,
            self.Insulin,
            self.BMI,
            self.DiabetesPedigreeFunction,
            self.Age,
            self.Urea,
            self.Cr,
            self.HbA1c,
            self.Chol,
            self.TG,
            self.HDL,
            self.LDL,
            self.VLDL,
            GENDER_MAPPING.get(self.Gender, None),
            BINARY_MAPPING.get(self.Polyuria, None),
            BINARY_MAPPING.get(self.Polydipsia, None),
            BINARY_MAPPING.get(self.SuddenWeightLoss, None),
            BINARY_MAPPING.get(self.Weakness, None),
            BINARY_MAPPING.get(self.Polyphagia, None),
            BINARY_MAPPING.get(self.GenitalThrush, None),
            BINARY_MAPPING.get(self.VisualBlurring, None),
            BINARY_MAPPING.get(self.Itching, None),
            BINARY_MAPPING.get(self.Irritability, None),
            BINARY_MAPPING.get(self.DelayedHealing, None),
            BINARY_MAPPING.get(self.PartialParesis, None),
            BINARY_MAPPING.get(self.MuscleStiffness, None),
            BINARY_MAPPING.get(self.Alopecia, None),
            BINARY_MAPPING.get(self.Obesity, None)
        ]
