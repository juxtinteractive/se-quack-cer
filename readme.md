# Se-quack-cer

A rubber duck driven step sequencer in Processing and PD

This project was written as fas as possible so the code is not particularly good. It's way more proof of concept than anything else.

Project works in Processing 2 and PD-extended 

The processing portion of this project depends on the following libraries

- [oscP5](http://www.sojamo.de/libraries/oscP5/)
- [OpenCV for Processing](https://github.com/atduskgreg/opencv-processing)
- [ControlP5](http://www.sojamo.de/libraries/controlP5/)

In order to run the processing sketch create a all white file named framemask.png in the ducks001 folder the same size as your web cam frame. The sketch will output a image called earlyframe.png that you can use a  a template to refine the mask (black is exuded white included). You may need to update the webcame name in the processing code. 

The pdSequencer patch will need 5 .wav files that for each of the tracks named 0.wav, 1.wav, 2.wav, 3.wav, 4.wav