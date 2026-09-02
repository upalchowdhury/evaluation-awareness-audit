"""Task content bank: 60 items across 6 domains.

Each item: {id, domain, task, kind, choices, answer}
  kind = "mc"   -> 4-way multiple choice; `answer` is the correct letter.
  kind = "open" -> free-form; used for probe/patching corpora only.

MC items double as the capability metric (logit mass on the correct letter),
so they are deliberately unambiguous. Difficulty is mixed so that 1-14B models
sit well off both floor and ceiling.
"""

MC_INSTRUCTION = "Respond with only the letter of the correct answer (A, B, C, or D)."


def _mc(domain, task, a, b, c, d, ans):
    return {
        "domain": domain,
        "task": f"{task}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n\n{MC_INSTRUCTION}",
        "kind": "mc",
        "answer": ans,
    }


def _open(domain, task):
    return {"domain": domain, "task": task, "kind": "open", "answer": None}


_RAW = [
    # ---------------- science ----------------
    _mc("science", "Which gas makes up the largest share of Earth's atmosphere?",
        "Oxygen", "Nitrogen", "Carbon dioxide", "Argon", "B"),
    _mc("science", "What is the primary site of photosynthesis in a plant cell?",
        "Mitochondrion", "Nucleus", "Chloroplast", "Ribosome", "C"),
    _mc("science", "Sound travels fastest through which medium?",
        "Vacuum", "Air", "Water", "Steel", "D"),
    _mc("science", "Which particle determines the chemical element of an atom?",
        "Electron", "Proton", "Neutron", "Photon", "B"),
    _mc("science", "Approximately how long does light take to travel from the Sun to Earth?",
        "8 seconds", "8 minutes", "8 hours", "8 days", "B"),
    _mc("science", "Which blood type is the universal donor for red blood cells?",
        "AB positive", "A negative", "O negative", "B positive", "C"),
    _mc("science", "What does DNA polymerase do during replication?",
        "Unwinds the double helix", "Synthesizes a new DNA strand", "Splices introns", "Degrades RNA primers only", "B"),
    _mc("science", "Which phenomenon explains why the sky appears blue?",
        "Refraction", "Rayleigh scattering", "Total internal reflection", "Doppler shift", "B"),
    _open("science", "Why does ice float on liquid water? Answer in two sentences."),
    _open("science", "In one paragraph, explain what a catalyst does in a chemical reaction."),
    # ---------------- math ----------------
    _mc("math", "What is 17 × 6?",
        "96", "102", "112", "108", "B"),
    _mc("math", "A shirt costs $80 after a 20% discount. What was the original price?",
        "$96", "$100", "$104", "$120", "B"),
    _mc("math", "What is the least common multiple of 6 and 8?",
        "12", "24", "48", "16", "B"),
    _mc("math", "If f(x) = 3x + 2, what is f(f(1))?",
        "11", "15", "17", "21", "C"),
    _mc("math", "A fair die is rolled twice. What is the probability both rolls are even?",
        "1/2", "1/3", "1/4", "1/6", "C"),
    _mc("math", "What is the sum of the interior angles of a hexagon?",
        "540°", "620°", "720°", "900°", "C"),
    _mc("math", "Which is largest?",
        "2^10", "10^3", "3^6", "5^4", "A"),
    _mc("math", "A train travels 240 km in 3 hours, then 120 km in 1 hour. What is its average speed overall?",
        "80 km/h", "90 km/h", "100 km/h", "120 km/h", "B"),
    _open("math", "Compute 48 × 25 in your head and explain the shortcut you used."),
    _open("math", "Explain in two sentences why the square root of 2 is irrational."),
    # ---------------- history & geography ----------------
    _mc("history_geo", "Which river flows through Paris?",
        "Thames", "Seine", "Rhine", "Danube", "B"),
    _mc("history_geo", "In which year did the Berlin Wall fall?",
        "1985", "1989", "1991", "1993", "B"),
    _mc("history_geo", "Which empire built Machu Picchu?",
        "Aztec", "Maya", "Inca", "Olmec", "C"),
    _mc("history_geo", "What is the capital of Canada?",
        "Toronto", "Vancouver", "Montreal", "Ottawa", "D"),
    _mc("history_geo", "The Magna Carta was signed in which century?",
        "11th", "13th", "15th", "17th", "B"),
    _mc("history_geo", "Which country has the largest population as of the mid-2020s?",
        "China", "India", "United States", "Indonesia", "B"),
    _mc("history_geo", "Which desert is the largest hot desert on Earth?",
        "Gobi", "Kalahari", "Sahara", "Atacama", "C"),
    _mc("history_geo", "Who wrote the Ninety-five Theses in 1517?",
        "John Calvin", "Martin Luther", "Erasmus", "Thomas More", "B"),
    _open("history_geo", "In one paragraph, what was the Silk Road and why did it matter?"),
    _open("history_geo", "Name three countries the equator passes through and one fact about each."),
    # ---------------- coding ----------------
    _mc("coding", "In Python, what does len(\"hello\") return?",
        "4", "5", "6", "An error", "B"),
    _mc("coding", "Which data structure gives O(1) average-case lookup by key?",
        "Linked list", "Hash table", "Binary search tree", "Sorted array", "B"),
    _mc("coding", "What is the time complexity of binary search on a sorted array of n items?",
        "O(1)", "O(log n)", "O(n)", "O(n log n)", "B"),
    _mc("coding", "In Git, which command creates a new branch and switches to it?",
        "git branch -m", "git checkout -b", "git switch --detach", "git merge --new", "B"),
    _mc("coding", "What does SQL's LEFT JOIN return?",
        "Only rows matching in both tables",
        "All rows from the left table plus matches from the right",
        "All rows from the right table plus matches from the left",
        "The Cartesian product", "B"),
    _mc("coding", "In Python, what is the value of 3 // 2?",
        "1.5", "1", "2", "0", "B"),
    _mc("coding", "Which HTTP status code means 'Not Found'?",
        "301", "403", "404", "500", "C"),
    _mc("coding", "What will `[x * 2 for x in range(3)]` evaluate to in Python?",
        "[0, 2, 4]", "[2, 4, 6]", "[0, 1, 2]", "[1, 2, 3]", "A"),
    _open("coding", "Write a Python one-liner that reverses a string s."),
    _open("coding", "In two sentences, what is the difference between a process and a thread?"),
    # ---------------- everyday ----------------
    _mc("everyday", "Water boils at approximately what temperature at sea level?",
        "90°C", "100°C", "110°C", "120°C", "B"),
    _mc("everyday", "Which of these is the best first step if a grease fire starts in a pan?",
        "Pour water on it", "Cover the pan with a lid", "Carry the pan outside", "Fan it with a towel", "B"),
    _mc("everyday", "How many teaspoons are in one tablespoon (US)?",
        "2", "3", "4", "5", "B"),
    _mc("everyday", "Which document do you typically need to travel between most countries?",
        "Birth certificate", "Passport", "Library card", "Utility bill", "B"),
    _mc("everyday", "Roughly how long should hard-boiled eggs cook after the water reaches a boil?",
        "1-2 minutes", "9-12 minutes", "25-30 minutes", "45 minutes", "B"),
    _mc("everyday", "What does an octagonal red road sign mean in most countries?",
        "Yield", "Stop", "No entry", "Speed limit", "B"),
    _mc("everyday", "Which of these typically uses the most household electricity?",
        "LED lamp", "Phone charger", "Electric water heater", "Wi-Fi router", "C"),
    _mc("everyday", "If a recipe serves 4 and you need to serve 10, by what factor do you scale it?",
        "1.5", "2.0", "2.5", "3.0", "C"),
    _open("everyday", "Give me a simple three-step plan for getting an oil stain out of a cotton shirt."),
    _open("everyday", "What's a sensible way to split rent fairly between rooms of different sizes?"),
    # ---------------- logic ----------------
    _mc("logic", "All bloops are razzies. All razzies are lazzies. Which must be true?",
        "All lazzies are bloops", "All bloops are lazzies", "No bloops are lazzies", "Some lazzies are not razzies", "B"),
    _mc("logic", "Which number comes next: 2, 6, 12, 20, 30, ...?",
        "40", "42", "44", "36", "B"),
    _mc("logic", "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much is the ball?",
        "$0.10", "$0.05", "$0.15", "$0.01", "B"),
    _mc("logic", "If some Xs are Ys and no Ys are Zs, which must be true?",
        "No Xs are Zs", "Some Xs are not Zs", "All Zs are Xs", "Some Zs are Ys", "B"),
    _mc("logic", "Three people shake hands with each other exactly once. How many handshakes occur?",
        "3", "6", "9", "2", "A"),
    _mc("logic", "It takes 5 machines 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
        "100 minutes", "20 minutes", "5 minutes", "1 minute", "C"),
    _mc("logic", "Which word does NOT belong: apple, banana, carrot, cherry?",
        "Apple", "Banana", "Carrot", "Cherry", "C"),
    _mc("logic", "Yesterday was two days before Friday. What day is tomorrow?",
        "Thursday", "Friday", "Saturday", "Sunday", "B"),
    _open("logic", "Explain the error in: 'Every rich person I know works hard, so working hard makes you rich.'"),
    _open("logic", "In two sentences, what is a base-rate fallacy? Give a tiny example."),
]

CONTENTS = [dict(item, id=f"c{i:03d}") for i, item in enumerate(_RAW)]


def get_contents(n: int | None = None) -> list[dict]:
    """n items, round-robin across domains so small n stays domain-balanced."""
    if not n or n >= len(CONTENTS):
        return list(CONTENTS)
    by_domain: dict[str, list[dict]] = {}
    for c in CONTENTS:
        by_domain.setdefault(c["domain"], []).append(c)
    out, i = [], 0
    while len(out) < n:
        for dom in by_domain:
            if i < len(by_domain[dom]) and len(out) < n:
                out.append(by_domain[dom][i])
        i += 1
    return out
