import replicate

input = {
    "text": "Hi! I'm Kokoro, a text-to-speech voice crafted by hexgrad — based on StyleTTS2. You can also find me in Kuluko, an app that lets you create fully personalized audiobooks — from characters to storylines — all tailored to your preferences. Want to give it a go? Search for Kuluko on the Apple or Android app store and start crafting your own story today!",
    "voice": "af_jessica"
}

output = replicate.run(
    "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
    input=input
)
with open("output.wav", "wb") as file:
    file.write(output.read())