# services/kami_brain.py
import logging
from purpleParrotMultiCharacterVoiceAssistant.config.personas import PersonaSuite, PersonaTheme

logger = logging.getLogger("kami_brain")

class KamiBrain:
    def __init__(self, default_persona: str = "kami"):
        self.active_persona_name = default_persona.lower()
        self.active_persona: PersonaTheme = PersonaSuite.get_persona(self.active_persona_name)
        # Persistent memory context per tracking session
        self.dialogue_history = []

    def switch_persona(self, persona_name: str) -> PersonaTheme:
        """
        Hot-swaps the underlying conversational identity matrix instantly.
        """
        logger.info(f"Mutating active voice persona state from {self.active_persona_name} to {persona_name}")
        self.active_persona_name = persona_name.lower()
        self.active_persona = PersonaSuite.get_persona(self.active_persona_name)
        return self.active_persona

    def compile_system_instructions(self, patient_context: dict = None) -> str:
        """
        Dynamically stitches together the core SLT system prompt using the active 
        persona's exact dialogue metrics from personas.py.
        """
        p = self.active_persona
        
        # Pull standard clinical constraints if provided (Age profiling out of System 1.1/2.1)
        age_tier = patient_context.get("age_tier", "Child") if patient_context else "Child"
        target_disorder = patient_context.get("target_disorder", "General Articulation") if patient_context else "General Articulation"

        base_prompt = (
            f"SYSTEM ROLE DEFINITION:\n"
            f"You are operating as the conversational voice assistant element of the Purple Parrot OS.\n"
            f"Your current identity skin is: {p.name}. You present yourself as a {p.animal_or_character}.\n\n"
            f"VOCAL DESIGN PARAMETERS:\n"
            f"- Your voice output model is set to: {p.vocal_profile}.\n"
            f"- Adhere closely to this conversational posture: {p.dialogue_tonality}.\n\n"
            f"CLINICAL BOUNDARIES & CONTEXT:\n"
            f"- Target Audience Profile: {age_tier} learner.\n"
            f"- Focus Therapeutic Area: {target_disorder}.\n\n"
            f"EXECUTION RULES:\n"
            f"1. Keep answers brief, clear, and perfectly adjusted for real-time speech processing loops.\n"
            f"2. Never break character. Fully project the style, warmth, quirks, or unique pacing of your active persona.\n"
            f"3. Incorporate subtle phonetic practice cues relevant to your tone when safe."
        )
        return base_prompt

    def append_interaction(self, role: str, text: str):
        self.dialogue_history.append({"role": role, "content": text})
        # Keep sliding memory window small to preserve lightning-fast processing
        if len(self.dialogue_history) > 20:
            self.dialogue_history.pop(0)