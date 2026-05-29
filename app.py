import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
import fitz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import re
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import spacy
import streamlit as st

df = pd.read_csv("resume_data.csv")


df_cleaned = df.dropna(thresh=len(df) * 0.5, axis=1)
columns_to_remove = ['result_types', 'company_urls', 'educational_results','professional_company_names','locations']
df_cleaned = df_cleaned.drop(columns=columns_to_remove)


def extract_first_year(passing_year):
    if isinstance(passing_year, str):
        years = passing_year.split(',') 
        return years[0].strip() 
    elif isinstance(passing_year, list):
        return passing_year[0]
    else:
        return passing_year 
    
def clean_passing_year(passing_year):

    if isinstance(passing_year, str):

        passing_year = passing_year.replace('[', '').replace(']', '').replace("'", "").strip()

        years = passing_year.split(',')
        return years[0].strip()
    else:
       return passing_year
    
df_cleaned['passing_years'] = df['passing_years'].apply(extract_first_year)
df_cleaned['passing_years'] = df['passing_years'].apply(clean_passing_year)
df_cleaned['passing_years'] = pd.to_numeric(df_cleaned['passing_years'], errors='coerce')

def calculate_age(passing_year):
    if pd.isnull(passing_year):
        return np.nan
  
    return (2026 - passing_year) + 22

df_cleaned['passing_years'] = df_cleaned['passing_years'].apply(calculate_age)
mean_value = df_cleaned['passing_years'].mean()
df_cleaned['passing_years'].fillna(abs(int(mean_value)), inplace=True)

########################################################
df_cleaned = df_cleaned.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x.fillna(x.mean()), axis=0)


vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
skills_vectorized = vectorizer.fit_transform(df_cleaned['skills'].fillna('')) 
skills_df = pd.DataFrame(skills_vectorized.toarray(), columns=vectorizer.get_feature_names_out())


skills_required_vectorized = vectorizer.fit_transform(df_cleaned['skills_required'].fillna('')) 
skills_required_df = pd.DataFrame(skills_required_vectorized.toarray(), columns=vectorizer.get_feature_names_out())


def get_all_years(date_input, default_current=2026):
    """دالة لتحويل القائمة أو النص إلى قائمة سنوات"""
    years = []
    
    # تحويل المدخل إلى قائمة حقيقية إذا كان نصاً يمثل قائمة
    if isinstance(date_input, str) and date_input.startswith('['):
        try:
            date_list = ast.literal_eval(date_input)
        except:
            date_list = [date_input]
    elif isinstance(date_input, list):
        date_list = date_input
    else:
        date_list = [date_input]

    for item in date_list:
        if pd.isna(item): continue
        
        item_str = str(item).strip()
        # التعامل مع الكلمات التي تعني الوقت الحالي
        if any(kw in item_str for kw in ["Ongoing", "Till Date", "Current", "Present"]):
            years.append(default_current)
        else:
            try:
                # تحويل النص لتاريخ واستخراج السنة
                dt = pd.to_datetime(item_str, errors='coerce')
                if pd.notnull(dt):
                    years.append(dt.year)
            except:
                continue
    return years

def calculate_total_experience(row):
    # الحصول على كل السنوات المذكورة في البداية والنهاية
    start_years = get_all_years(row['start_dates'])
    end_years = get_all_years(row['end_dates'])
    
    if not start_years:
        return 0
    
    # أقدم تاريخ بدأت فيه (أصغر سنة)
    min_start = min(start_years)
    
    # أحدث تاريخ انتهيت فيه (أكبر سنة)
    # لو قائمة النهاية فاضية بنفترض إنه شغال لحد دلوقتي
    max_end = max(end_years) if end_years else 2026
    
    diff = max_end - min_start
    return max(1, diff)

# تطبيق الدالة الجديدة
df_cleaned['start_dates'] = df_cleaned.apply(calculate_total_experience, axis=1)
df_cleaned.rename(columns={'start_dates': 'experiencere'}, inplace=True)
df_cleaned.rename(columns={'passing_years': 'age'}, inplace=True)

columns_to_remove = ['end_dates']
df_cleaned = df_cleaned.drop(columns=columns_to_remove)


def extract_average_value(text):
    if pd.isna(text) or text == "" or str(text).lower() == 'nan':
        return np.nan
    
    # تنظيف النص من الأقواس والعلامات
    text = str(text).replace('[', '').replace(']', '').replace("'", "")
    
    # البحث عن كل الأرقام في النص
    numbers = re.findall(r'\d+', text)
    numbers = [int(n) for n in numbers]
    
    if len(numbers) >= 2:
        # لو لقيت رقمين (زي 25 و 40) هات المتوسط
        return sum(numbers[:2]) / 2
    elif len(numbers) == 1:
        # لو رقم واحد (زي at least 5) هاته هو
        return float(numbers[0])
    
    return np.nan

# تطبيق الدالة على الأعمدة المطلوبة
df_cleaned['age_requirement'] = df_cleaned['age_requirement'].apply(extract_average_value)
df_cleaned['experiencere_requirement'] = df_cleaned['experiencere_requirement'].apply(extract_average_value)

df_cleaned['age_requirement'] = df_cleaned['age_requirement'].round().astype('Int64')
df_cleaned['experiencere_requirement'] = df_cleaned['experiencere_requirement'].round().astype('Int64')
df_cleaned['age'] = df_cleaned['age'].round().astype('Int64')
df_cleaned['experiencere'] = df_cleaned['experiencere'].round().astype('Int64')



def deep_clean_text(text):
    if pd.isna(text) or text == "": return ""
    # 1. إزالة الأقواس المربعة والعلامات الخاصة بالـ Lists
    text = str(text).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    # 2. تحويل لـ lowercase وإزالة علامات الترقيم
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    # 3. إزالة الـ Stop words والمسافات الزائدة
    words = text.split()
    clean_words = [w for w in words if w not in stop_words]
    return " ".join(clean_words)

# تطبيق التنظيف على كل الأعمدة النصية مرة واحدة
text_columns = ['related_skils_in_job', 'responsibilities', 'positions', 'major_field_of_studies', 'educationaL_requirements','degree_names','educational_institution_name']
for col in text_columns:
    df_cleaned[col] = df_cleaned[col].apply(deep_clean_text)


    df_cleaned['all_resume_text'] = (
    df_cleaned['positions'].astype(str) + " " + 
    df_cleaned['related_skils_in_job'].astype(str) + " " + 
    df_cleaned['major_field_of_studies'].astype(str) + " " + 
    df_cleaned['skills'].astype(str)
)

df_cleaned['all_job_text'] = (
    df_cleaned['educationaL_requirements'].astype(str) + " " + 
    df_cleaned['responsibilities'].astype(str) + " " + 
    df_cleaned['skills_required'].astype(str)
)


tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(pd.concat([df_cleaned['all_resume_text'], df_cleaned['all_job_text']]))


mid = len(df_cleaned)
resume_vectors = tfidf_matrix[:mid]
job_vectors = tfidf_matrix[mid:]


similarities = [cosine_similarity(resume_vectors[i], job_vectors[i])[0][0] for i in range(mid)]
df_cleaned['text_similarity_score'] = similarities


df_cleaned['age'] = df_cleaned['age'].fillna(df_cleaned['age'].mean())
df_cleaned['experiencere'] = df_cleaned['experiencere'].fillna(0)
df_cleaned['age_requirement'] = df_cleaned['age_requirement'].fillna(df_cleaned['age_requirement'].mean())
df_cleaned['experiencere_requirement'] = df_cleaned['experiencere_requirement'].fillna(0)

# حساب الفروقات (دي ميزات قوية جداً للموديل)
df_cleaned['exp_diff'] = df_cleaned['experiencere'] - df_cleaned['experiencere_requirement']
df_cleaned['age_diff'] = df_cleaned['age'] - df_cleaned['age_requirement']


X = df_cleaned[['text_similarity_score', 'age', 'experiencere', 'exp_diff', 'age_diff']]
y = df_cleaned['matched_score'] # السكور الحقيقي اللي في الداتا

# 2. تقسيم الداتا
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. تدريب الموديل
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. إضافة التوقع النهائي للداتا فريم
df_cleaned['final_match_rank'] = rf_model.predict(X)

ranked_candidates = df_cleaned.sort_values(by='final_match_rank', ascending=False)




def extract_smart_cv_data(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = " ".join([page.get_text() for page in doc])
    current_year = 2026

    # أ- حساب السن (ميلاد أو تخرج + 22)
    final_age = 25
    dob_match = re.search(r'(?:birth|born|dob|date of birth)\D+(\d{4})', text, re.I)
    edu_years = re.findall(r'(?:20[0-2][0-9]|19[7-9][0-9])(?=.*(?:University|College|Bachelor|BSc|Education))', text, re.I)
    
    if dob_match:
        final_age = current_year - int(dob_match.group(1))
    elif edu_years:
        final_age = (current_year - int(edu_years[0])) + 22
    else:
        all_years = sorted(list(set([int(y) for y in re.findall(r'\b(19[7-9][0-9]|20[0-2][0-9])\b', text)])))
        if all_years:
            final_age = (current_year - all_years[0]) + 22

    # ب- حساب الخبرة (جمل صريحة أو فترات)
    exp_matches = re.findall(r'(\d+)\s?\+?\s?years?\s?(?:of\s?)?(?:experience|exp)', text, re.I)
    if exp_matches:
        final_exp = int(exp_matches[0])
    else:
        all_years = sorted(list(set([int(y) for y in re.findall(r'\b(19[7-9][0-9]|20[0-2][0-9])\b', text)])))
        final_exp = max(all_years) - min(all_years) if len(all_years) >= 2 else 0

    # ج- استخراج البلوكات (المهارات والتعليم)
    skills_sec = re.search(r'(?:skills|technologies)\s?[:\-]?\n?(.*?)(?:\n\n|Experience|Education|$)', text, re.I | re.DOTALL)
    edu_sec = re.search(r'(?:education|academic|courses)\s?[:\-]?\n?(.*?)(?:\n\n|Skills|Experience|$)', text, re.I | re.DOTALL)

    return {
        "age": final_age,
        "experiencere": final_exp,
        "skills": skills_sec.group(1).strip() if skills_sec else "Check CV text",
        "education": edu_sec.group(1).strip() if edu_sec else "Check CV text",
        "full_text": text
    }

# --- 2. واجهة Streamlit ---
st.set_page_config(page_title="AI Recruitment", layout="wide")
st.title("🚀 AI Job-CV Matcher")

tab1, tab2 = st.tabs(["📝 Job Posting", "📄 Upload & Match"])

with tab1:
    st.header("Job Details")
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Position", "Android Developer")
        job_age = st.number_input("Target Age", 30)
    with col2:
        job_exp = st.number_input("Required Experience", 5)
        job_skills = st.text_input("Must-have Skills", "Java, Python, Kotlin")
    job_desc = st.text_area("Full Description", height=150)
    
    if st.button("Save Posting"):
        st.session_state['job'] = {"age": job_age, "exp": job_exp, "desc": job_desc}
        st.success("Saved!")

with tab2:
    if 'job' not in st.session_state:
        st.warning("Please fill Job Posting first.")
    else:
        pdf = st.file_uploader("Upload CV", type="pdf")
        if pdf:
            cv = extract_smart_cv_data(pdf)
            st.subheader("Confirm Data")
            c_a, c_e = st.columns(2)
            # هنا التعديل اليدوي اللي طلبته
            final_age = c_a.number_input("Age", value=int(cv['age']))
            final_exp = c_e.number_input("Experience", value=int(cv['experiencere']))
            
            if st.button("Calculate Match"):
                # منطق الحسبة الذكية (Weights)
                tfidf = TfidfVectorizer(stop_words='english')
                vecs = tfidf.fit_transform([st.session_state['job']['desc'], cv['full_text']])
                text_sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]

                # سكور السن (قرب من الـ Average)
                age_score = max(0, 100 - (abs(final_age - st.session_state['job']['age']) * 3))
                
                # سكور الخبرة (Bonus لو أكتر)
                req = st.session_state['job']['exp']
                exp_score = 100 + ((final_exp - req) * 5) if final_exp >= req else 100 - ((req - final_exp) * 15)
                
                final_pct = (text_sim * 50) + (age_score * 0.20) + (exp_score * 0.30)
                st.metric("Compatibility Score", f"{min(100.0, final_pct):.1f}%")
                st.balloons()