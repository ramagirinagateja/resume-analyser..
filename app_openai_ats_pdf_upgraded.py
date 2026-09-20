import io
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from xml.sax.saxutils import escape


# 1. Define the structured output format using Pydantic
class ResumeChange(BaseModel):
    section: str = Field(description="Resume section changed, e.g. Summary, Skills, Experience")
    original: str = Field(description="Original resume text that was changed")
    updated: str = Field(description="Updated resume text")
    reason: str = Field(description="Why the change improves alignment with the job description")

class ResumeAnalysis(BaseModel):
    match_percentage: int = Field(
        description="ATS match score from 0 to 100 based on the job description"
    )
    matched_skills: list[str] = Field(
        description="Keywords and skills found in the resume that match the job description"
    )
    missing_skills: list[str] = Field(
        description="Important keywords and skills from the job description missing in the resume"
    )
    recommendations: list[str] = Field(
        description="Actionable bullet points to improve the resume for this job role"
    )
    updated_resume: str = Field(
        description="Complete updated resume. Preserve facts from the original resume. Do not invent experience, education, projects, skills, metrics, certifications, or achievements."
    )
    changes: list[ResumeChange] = Field(
        description="Every meaningful change made between the original and updated resume"
    )


def create_resume_pdf(resume_text: str) -> bytes:
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=8)
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=7, spaceAfter=3)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=3)
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=15*mm, bottomMargin=15*mm, title="Optimized Resume")
    story=[]; first=True
    headings={"summary","professional summary","objective","skills","technical skills","experience","professional experience","education","certifications","projects","achievements","internships","languages","awards"}
    for raw in resume_text.splitlines():
        line=raw.strip()
        if not line: story.append(Spacer(1,4)); continue
        if line.startswith(("-","*","•")): story.append(Paragraph("• "+escape(line[1:].strip()), body))
        elif first: story.append(Paragraph(escape(line), title))
        elif line.lower().rstrip(":") in headings or line.endswith(":") or line.isupper(): story.append(Paragraph(escape(line), heading))
        else: story.append(Paragraph(escape(line), body))
        first=False
    doc.build(story)
    return buffer.getvalue()

# Streamlit page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer by Venkys.AI")
st.write("Upload your resume and job description to analyze ATS compatibility, generate a truthful tailored resume, and see exactly what changed.")


# Sidebar for API Key input
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password"
    )
    st.markdown(
        "[Create/manage your OpenAI API key](https://platform.openai.com/api-keys)"
    )


# Main Interface: Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF format only)",
        type=["pdf"]
    )

with col2:
    st.subheader("2. Job Description")
    job_description = st.text_area(
        "Paste the Target Job Description Here",
        height=200
    )


# Process analysis when button is clicked
if st.button("Analyze Resume", type="primary"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Extracting text and analyzing with OpenAI..."):
            try:
                # Extract text from PDF
                reader = PdfReader(uploaded_file)
                resume_text = ""

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text + "\n"

                if not resume_text.strip():
                    st.error(
                        "Could not extract text from this PDF. "
                        "Please upload a text-based PDF."
                    )
                    st.stop()

                # Initialize OpenAI model through LangChain
                # GPT-5.6 Luna is selected here as a cost-sensitive model.
                llm = ChatOpenAI(
                    model="gpt-5.6-luna",
                    api_key=openai_api_key,
                    temperature=0.2
                )

                # Structured JSON parser
                parser = JsonOutputParser(
                    pydantic_object=ResumeAnalysis
                )

                # Prompt
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are an expert ATS (Applicant Tracking System) "
                        "resume optimizer. Compare the candidate's resume against "
                        "the job description and create a tailored version. "
                        "Improve wording, ordering, clarity, keyword alignment, "
                        "and professional phrasing where supported by the original resume. "
                        "Never invent qualifications, experience, skills, projects, "
                        "metrics, employers, dates, certifications, or achievements. "
                        "If a job requirement is genuinely missing, list it as missing "
                        "instead of falsely adding it. The updated resume must remain "
                        "truthful and ATS-friendly. Return output strictly in JSON format "
                        "matching the schema instructions.\n{format_instructions}"
                    ),
                    (
                        "human",
                        "RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}"
                    )
                ])

                # Chain: Prompt -> OpenAI -> JSON Parser
                chain = prompt | llm | parser

                # Execute
                result = chain.invoke({
                    "resume": resume_text,
                    "job_description": job_description,
                    "format_instructions": parser.get_format_instructions()
                })

                # Render results
                st.success("Analysis Complete!")
                st.divider()

                score = result.get("match_percentage", 0)

                if score >= 75:
                    st.balloons()
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Strong Match"
                    )
                elif score >= 50:
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Needs Improvement",
                        delta_color="off"
                    )
                else:
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Weak Match",
                        delta_color="inverse"
                    )

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.subheader("✅ Matched Skills & Keywords")
                    for skill in result.get("matched_skills", []):
                        st.markdown(f"- {skill}")

                with res_col2:
                    st.subheader("❌ Missing Critical Keywords")
                    for skill in result.get("missing_skills", []):
                        st.markdown(
                            f"- <span style='color:#ff4b4b'>"
                            f"**{skill}**</span>",
                            unsafe_allow_html=True
                        )

                st.subheader("💡 Recommendations to Optimize Your Resume")
                for rec in result.get("recommendations", []):
                    st.markdown(f"* {rec}")

                st.divider()

                # Tailored resume
                st.subheader("📝 Updated Resume")
                st.caption(
                    "This version is tailored to the job description using only information "
                    "supported by your original resume."
                )
                updated_resume = st.text_area("Edit your optimized resume before downloading", value=result.get("updated_resume", ""), height=550, key="editable_resume")
                if updated_resume.strip():
                    st.download_button("⬇️ Download Optimized Resume as PDF", data=create_resume_pdf(updated_resume), file_name="optimized_resume.pdf", mime="application/pdf", type="primary")
                    st.download_button("⬇️ Download Resume as TXT", data=updated_resume, file_name="optimized_resume.txt", mime="text/plain")

                # Transparent change log
                st.subheader("🔍 What Changed & Why")
                changes = result.get("changes", [])
                if not changes:
                    st.info("No meaningful changes were detected.")
                else:
                    for i, change in enumerate(changes, 1):
                        with st.expander(
                            f"{i}. {change.get('section', 'Resume section')} — "
                            f"Why this changed"
                        ):
                            st.markdown("**Original:**")
                            st.code(change.get("original", ""), language=None)
                            st.markdown("**Updated:**")
                            st.code(change.get("updated", ""), language=None)
                            st.markdown(
                                f"**Reason:** {change.get('reason', '')}"
                            )


            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info(
                    "If the error mentions billing, quota, or insufficient "
                    "credits, check your OpenAI API billing/usage settings."
                )
