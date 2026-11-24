import re, os, sys, shutil, subprocess
sys.path.insert(0,'/home/claude/kdp'); sys.path.insert(0,'/home/claude/build')
import eq2png, bookkit as bk
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset']='stix'
A='/home/claude/art/pkg/'; B='/home/claude/art/pkgB/'
shutil.rmtree(B,ignore_errors=True); shutil.copytree(A,B)
d=open(B+'word/document.xml',encoding='utf8').read(); eqs=eq2png.equations(d); os.makedirs('/home/claude/art/eq',exist_ok=True)
rl=open(B+'word/_rels/document.xml.rels',encoding='utf8').read(); add=''; out=[]; pos=0
for i,(s,e,tex) in enumerate(eqs):
    p=f'/home/claude/art/eq/eq{i:02d}.png'; eq2png.render(tex,p); shutil.copy(p,B+f'word/media/aeq{i:02d}.png'); w,h=Image.open(p).size; cx=int(w/450*914400); cy=int(h/450*914400); rid=f'rIdAEq{i:02d}'; did=7000+i
    add+=f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/aeq{i:02d}.png"/>'
    run=(f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{did}" name="Equation {i}" descr="Equation"/><wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{did}" name="aeq{i:02d}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
    out.append(d[pos:s]+run); pos=e
d=''.join(out)+d[pos:]; open(B+'word/document.xml','w',encoding='utf8').write(d); open(B+'word/_rels/document.xml.rels','w',encoding='utf8').write(rl.replace('</Relationships>',add+'</Relationships>'))
V='/home/claude/art/Horizon_Transform_Article_pdfsource.docx'
if os.path.exists(V): os.remove(V)
subprocess.run(f"cd {B} && zip -Xqr -D {V} '[Content_Types].xml' _rels docProps word",shell=True,check=True)
r='/home/claude/art/render'; shutil.rmtree(r,ignore_errors=True); os.makedirs(r); shutil.copy(V,r)
subprocess.run(f'cd {r} && python3 /mnt/skills/public/docx/scripts/office/soffice.py --headless --convert-to pdf {os.path.basename(V)} >/dev/null 2>&1',shell=True)
print(subprocess.run(['pdfinfo',r+'/Horizon_Transform_Article_pdfsource.pdf'],capture_output=True,text=True).stdout.split('Pages:')[1].split('\n')[0].strip(),'pages; equations',len(eqs))
