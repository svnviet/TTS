import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from pydub import AudioSegment
sentences="""Số người nhiễm và chết vì nCoV trên toàn cầu tăng lên lần lượt khi có hàng nghìn ca nhiễm mới ở nhiều nước châu Âu   .lan rộng và diễn biến phức tạp khi xuất hiện tại quốc gia và vùng lãnh thổ. Nhiều quốc gia và vùng lãnh thổ ghi nhận các ca nhiễm nCoV đầu tiên như Benin, Greenland, Liberia, Somalia và Tanzania. Châu Âu trở thành tâm dịch toàn cầu, khi Trung Quốc, nơi khởi phát dịch, có số ca nhiễm mới và tử vong giảm mạnh.
Italy, vùng dịch lớn nhất châu Âu và lớn thứ hai trên thế giới, thêm ca nhiễm mới và trường hợp tử vong, nâng số ca nhiễm và người chết trên toàn quốc lên và . Tây Ban Nha, Đức và Pháp xuất hiện thêm hơn rường hợp dương tính, đưa số ca nhiễm lên lần lượt là ."""

def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path =r'D:\tacotron_tensorflow-master\tacotron\testckpt\e\\'
  for i, text in enumerate(sentences.split('.')):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))

#write full audio and fix speech
  Dir = os.listdir(base_path)
  sound2=AudioSegment.silent()
  for i in (Dir):
    sound = AudioSegment.from_wav(base_path+i) + AudioSegment.silent(duration=500)
    sound2 +=sound
  print("Make FUll audio : ...")
  
  speed=args.speed
  sound2 = sound2._spawn(sound2.raw_data, overrides={
                 "frame_rate": int(sound2.frame_rate * speed)})
  sound2 = sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)
  sound2.export("path10.wav", format="wav")
  return sound2

def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint',default=r'D:\tacotron_tensorflow-master\tacotron\testckpt\model.ckpt-152000')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--speed',type=float,default=1.0)
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
