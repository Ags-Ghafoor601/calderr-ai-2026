"""
AI-Powered Hiring Pipeline — Sample Data
==========================================
10 sample resumes + 2 job descriptions for demonstration.
Includes diverse candidates to test bias detection.
"""


# ---------------------------------------------------------------------------
# Job Descriptions
# ---------------------------------------------------------------------------

JOB_DESCRIPTIONS = [
    {
        "id": "JD-001",
        "title": "Senior Python Engineer",
        "department": "Engineering",
        "required_skills": [
            "Python", "FastAPI", "SQL", "REST APIs", "Git",
            "Docker", "Unit Testing",
        ],
        "preferred_skills": [
            "LangChain", "LangGraph", "Machine Learning",
            "Kubernetes", "AWS", "CI/CD",
        ],
        "min_experience": 4,
        "education_requirement": "Bachelor's in Computer Science or related",
        "description": (
            "We are looking for a Senior Python Engineer to join our AI Platform team. "
            "You will design and build production-grade LLM-powered workflows, "
            "maintain our FastAPI microservices, and mentor junior engineers. "
            "Experience with agentic AI frameworks is a strong plus."
        ),
        "salary_range": "$130,000 – $170,000",
    },
    {
        "id": "JD-002",
        "title": "ML Engineer — NLP",
        "department": "Data Science",
        "required_skills": [
            "Python", "PyTorch", "Transformers", "NLP",
            "Hugging Face", "SQL",
        ],
        "preferred_skills": [
            "LangChain", "RAG", "Vector Databases",
            "Fine-tuning", "MLOps", "ONNX",
        ],
        "min_experience": 3,
        "education_requirement": "Master's in CS, ML, or related",
        "description": (
            "Join our NLP team to build and deploy transformer-based models. "
            "You will work on RAG pipelines, fine-tune LLMs, and build "
            "evaluation frameworks for production AI systems."
        ),
        "salary_range": "$140,000 – $180,000",
    },
]


# ---------------------------------------------------------------------------
# Sample Resumes (10 candidates — diverse backgrounds)
# ---------------------------------------------------------------------------

SAMPLE_RESUMES = [
    # --- Candidate 1: Strong match for JD-001 ---
    {
        "name": "Amara Okafor",
        "email": "amara.okafor@email.com",
        "phone": "+1-555-1001",
        "years_experience": 6,
        "education": "B.Sc. Computer Science",
        "university": "University of Lagos",
        "skills": [
            "Python", "FastAPI", "Django", "SQL", "PostgreSQL",
            "Docker", "Kubernetes", "Git", "REST APIs",
            "LangChain", "LangGraph", "Unit Testing", "CI/CD",
        ],
        "previous_roles": [
            "Senior Backend Engineer at TechCorp (3 yrs)",
            "Python Developer at StartupXYZ (2 yrs)",
            "Junior Developer at CodeWorks (1 yr)",
        ],
        "summary": (
            "Senior Python engineer with 6 years of experience building "
            "scalable microservices and AI-powered platforms. Deep expertise "
            "in FastAPI, Docker, and agentic AI frameworks including LangChain "
            "and LangGraph. Passionate about clean architecture and mentoring."
        ),
    },
    # --- Candidate 2: Good match, less experience ---
    {
        "name": "Liam Chen",
        "email": "liam.chen@email.com",
        "phone": "+1-555-1002",
        "years_experience": 3,
        "education": "B.Sc. Software Engineering",
        "university": "University of Toronto",
        "skills": [
            "Python", "FastAPI", "Flask", "SQL", "MongoDB",
            "Docker", "Git", "REST APIs", "Unit Testing",
            "AWS Lambda",
        ],
        "previous_roles": [
            "Backend Developer at DataStream (2 yrs)",
            "Software Intern at CloudBase (1 yr)",
        ],
        "summary": (
            "Enthusiastic Python developer with 3 years of experience. "
            "Strong foundation in FastAPI and cloud deployment. "
            "Looking to grow into senior roles with AI/ML focus."
        ),
    },
    # --- Candidate 3: Strong match for JD-002 ---
    {
        "name": "Priya Sharma",
        "email": "priya.sharma@email.com",
        "phone": "+1-555-1003",
        "years_experience": 5,
        "education": "M.Sc. Machine Learning",
        "university": "Indian Institute of Technology Bombay",
        "skills": [
            "Python", "PyTorch", "TensorFlow", "Transformers",
            "NLP", "Hugging Face", "SQL", "RAG",
            "Vector Databases", "Fine-tuning", "MLOps",
        ],
        "previous_roles": [
            "ML Engineer at AILabs (2 yrs)",
            "NLP Researcher at DeepTech (2 yrs)",
            "Data Scientist at AnalyticsCo (1 yr)",
        ],
        "summary": (
            "ML engineer specializing in NLP and transformer architectures. "
            "5 years experience building production NLP systems, RAG pipelines, "
            "and fine-tuning large language models. Published 3 papers on "
            "efficient attention mechanisms."
        ),
    },
    # --- Candidate 4: Moderate match ---
    {
        "name": "James O'Brien",
        "email": "james.obrien@email.com",
        "phone": "+1-555-1004",
        "years_experience": 8,
        "education": "B.Sc. Information Technology",
        "university": "University of Melbourne",
        "skills": [
            "Java", "Spring Boot", "Python", "SQL",
            "Microservices", "Docker", "AWS", "Git",
            "Agile", "REST APIs",
        ],
        "previous_roles": [
            "Lead Engineer at FinServ Global (3 yrs)",
            "Senior Developer at BankTech (3 yrs)",
            "Java Developer at ConsultCorp (2 yrs)",
        ],
        "summary": (
            "Experienced lead engineer with 8 years in enterprise Java "
            "and microservices. Transitioning to Python and AI. Strong "
            "system design and team leadership skills."
        ),
    },
    # --- Candidate 5: Junior candidate ---
    {
        "name": "Sofia Rodriguez",
        "email": "sofia.rodriguez@email.com",
        "phone": "+1-555-1005",
        "years_experience": 1,
        "education": "B.Sc. Computer Science",
        "university": "Universidad de Buenos Aires",
        "skills": [
            "Python", "JavaScript", "React", "SQL",
            "Git", "HTML", "CSS",
        ],
        "previous_roles": [
            "Junior Developer at WebCraft (1 yr)",
        ],
        "summary": (
            "Recent graduate with 1 year of professional experience. "
            "Eager to learn and grow in backend development. "
            "Strong problem-solving skills from competitive programming."
        ),
    },
    # --- Candidate 6: Strong for JD-001, different background ---
    {
        "name": "Fatima Al-Hassan",
        "email": "fatima.alhassan@email.com",
        "phone": "+1-555-1006",
        "years_experience": 5,
        "education": "M.Sc. Computer Science",
        "university": "King Abdullah University of Science and Technology",
        "skills": [
            "Python", "FastAPI", "SQL", "Docker",
            "Kubernetes", "Git", "REST APIs",
            "Machine Learning", "LangChain", "Unit Testing",
            "CI/CD", "Terraform",
        ],
        "previous_roles": [
            "Platform Engineer at CloudAI (2 yrs)",
            "Backend Engineer at TechForward (2 yrs)",
            "Research Assistant at KAUST (1 yr)",
        ],
        "summary": (
            "Platform engineer with deep Python and infrastructure expertise. "
            "Built ML serving platforms handling 10K+ requests/sec. "
            "Experience with LangChain for production agentic systems."
        ),
    },
    # --- Candidate 7: Career changer ---
    {
        "name": "Michael Thompson",
        "email": "michael.thompson@email.com",
        "phone": "+1-555-1007",
        "years_experience": 2,
        "education": "Bootcamp Certificate",
        "university": "General Assembly",
        "skills": [
            "Python", "JavaScript", "React", "Node.js",
            "SQL", "Git", "Docker",
        ],
        "previous_roles": [
            "Full-Stack Developer at StartupHub (2 yrs)",
            "Marketing Manager at AdAgency (5 yrs — previous career)",
        ],
        "summary": (
            "Career changer from marketing to software engineering. "
            "2 years of professional development experience after "
            "completing a coding bootcamp. Quick learner with strong "
            "communication skills."
        ),
    },
    # --- Candidate 8: Strong NLP background ---
    {
        "name": "Yuki Tanaka",
        "email": "yuki.tanaka@email.com",
        "phone": "+1-555-1008",
        "years_experience": 4,
        "education": "Ph.D. Natural Language Processing",
        "university": "University of Tokyo",
        "skills": [
            "Python", "PyTorch", "Transformers", "NLP",
            "Hugging Face", "BERT", "GPT", "SQL",
            "RAG", "Vector Databases", "ONNX",
            "Fine-tuning", "Research",
        ],
        "previous_roles": [
            "NLP Research Scientist at LangAI (2 yrs)",
            "PhD Researcher at UTokyo NLP Lab (2 yrs)",
        ],
        "summary": (
            "NLP researcher turned engineer with expertise in transformer "
            "architectures and production NLP systems. Published 8 papers "
            "at ACL, EMNLP, and NeurIPS. Built RAG systems serving "
            "enterprise clients."
        ),
    },
    # --- Candidate 9: DevOps focus ---
    {
        "name": "Oleksandr Kovalenko",
        "email": "oleks.kovalenko@email.com",
        "phone": "+1-555-1009",
        "years_experience": 7,
        "education": "B.Sc. Computer Engineering",
        "university": "National Technical University of Ukraine",
        "skills": [
            "Python", "Bash", "Docker", "Kubernetes",
            "Terraform", "AWS", "GCP", "CI/CD",
            "Jenkins", "Prometheus", "Grafana",
        ],
        "previous_roles": [
            "Senior DevOps Engineer at ScaleUp (3 yrs)",
            "Infrastructure Engineer at HostPro (2 yrs)",
            "Systems Admin at NetCorp (2 yrs)",
        ],
        "summary": (
            "DevOps engineer with 7 years of infrastructure experience. "
            "Expert in containerization, orchestration, and cloud platforms. "
            "Some Python scripting for automation but limited application "
            "development experience."
        ),
    },
    # --- Candidate 10: Overqualified executive ---
    {
        "name": "Elizabeth Warren-Hughes",
        "email": "e.warren.hughes@email.com",
        "phone": "+1-555-1010",
        "years_experience": 15,
        "education": "MBA, M.Sc. Computer Science",
        "university": "Stanford University",
        "skills": [
            "Python", "Java", "System Design", "Architecture",
            "Leadership", "Strategy", "SQL", "AWS",
            "Machine Learning", "Product Management",
        ],
        "previous_roles": [
            "VP of Engineering at MegaTech (4 yrs)",
            "Director of AI at InnovateCo (3 yrs)",
            "Senior Architect at Enterprise Systems (4 yrs)",
            "Software Engineer at Google (4 yrs)",
        ],
        "summary": (
            "Engineering executive with 15 years spanning hands-on "
            "engineering to VP-level leadership. Deep technical background "
            "in distributed systems and ML. Looking to return to an "
            "individual contributor role."
        ),
    },
]
