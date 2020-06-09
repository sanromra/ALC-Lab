""" emoticon recognition via patterns.  tested on english-language twitter, but
probably works for other social media dialects. """

__author__ = "Brendan O'Connor (anyall.org, brenocon@gmail.com)"
__version__= "april 2009"

#from __future__ import print_function
import re,sys

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')

NormalEyes = r'[:=]'
Wink = r'[;]'

NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...

HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned

Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Happy_emojis = mycompile( "[😹😘💕😍👌😂💜💛💙🌚✨☺️🎊🎁🎂🎈🎉💪😻💖🙈💃😘🔥😁😎]")
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)
Sad_emojis = mycompile( "[🙁😒💔😫😢😡😕]")

Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
Tongue_emojis = mycompile("[😝]")
Other_RE = mycompile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths )

Emoticon = (
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea + 
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"|"+HappyMouths+")"
)
Emoticon_RE = mycompile(Emoticon)

#Emoticon_RE = "|".join([Happy_RE,Sad_RE,Wink_RE,Tongue_RE,Other_RE])
#Emoticon_RE = mycompile(Emoticon_RE)

def analyze_tweet(text):
  h= Happy_RE.search(text)
  he = Happy_emojis.search(text)
  s= Sad_RE.search(text)
  se= Sad_emojis.search(text)
  if (h or he) and (s or se): return "OTHER"
  if h or he: return "HAPPY"
  if s or se: return "SAD"
  return "OTHER"

  # more complex & harder, so disabled for now
  w= Wink_RE.search(text)
  t= Tongue_RE.search(text)
  te= Tongue_emojis.search(text)
  a= Other_RE.search(text)
  h,he,w,s,se,t,te,a = [bool(x) for x in [h,he,w,s,se,t,te,a]]
  if sum([h,he,w,s,se,t,te,a])>1: return "OTHER"
  if sum([h,he,w,s,se,t,te,a])==1:
    if h or he: return "HAPPY"
    if s or se: return "SAD"
    if w: return "WINK"
    if a: return "OTHER"
    if t or te: return "TONGUE"
  return "OTHER"

if __name__=='__main__':
  for line in sys.stdin:
    import sane_re
    sane_re._S(line[:-1]).show_match(Emoticon_RE, numbers=False)
    #print(analyze_tweet(line.strip()), line.strip(), sep="\t")