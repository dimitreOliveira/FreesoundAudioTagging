![](https://blog.freesound.org/wp-content/uploads/2017/12/updated_logo.png)

# Freesound General-Purpose Audio Tagging Challenge
### Can you automatically recognize sounds from a wide range of real-world environments?

link for the kaggle competition: https://www.kaggle.com/c/freesound-audio-tagging


### Competition Description
Some sounds are distinct and instantly recognizable, like a baby’s laugh or the strum of a guitar.

Other sounds aren’t clear and are difficult to pinpoint. If you close your eyes, can you tell which of the sounds below is a chainsaw versus a blender?

Moreover, we often experience a mix of sounds that create an ambience – like the clamoring of construction, a hum of traffic from outside the door, blended with loud laughter from the room, and the ticking of the clock on your wall. The sound clip below is of a busy food court in the UK.

Partly because of the vastness of sounds we experience, no reliable automatic general-purpose audio tagging systems exist. Currently, a lot of manual effort is required for tasks like annotating sound collections and providing captions for non-speech events in audiovisual content.

To tackle this problem, Freesound (an initiative by MTG-UPF that maintains a collaborative database with over 370,000 Creative Commons Licensed sounds) and Google Research’s Machine Perception Team (creators of AudioSet, a large-scale dataset of manually annotated audio events with over 500 classes) have teamed up to develop the dataset for this competition.

You’re challenged to build a general-purpose automatic audio tagging system using a dataset of audio files covering a wide range of real-world environments. Sounds in the dataset include things like musical instruments, human sounds, domestic sounds, and animals from Freesound’s library, annotated using a vocabulary of more than 40 labels from Google’s AudioSet ontology. To succeed in this competition your systems will need to be able to recognize an increased number of sound events of very diverse nature, and to leverage subsets of training data featuring annotations of varying reliability (see Data section for more information).