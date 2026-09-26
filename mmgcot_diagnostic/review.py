"""Build evaluation-only blinded visual packets and reconcile two reviews.

No generated box or condition name is included in an entity packet. Geometry
packets use opaque cell identifiers, with the key stored outside the packet.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from mmgcot_diagnostic.protocol import stable_seed

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def font(size):
    return ImageFont.truetype(FONT,size)


def multiline(canvas, text, xy, *, width=105, size=18, fill="black"):
    d=ImageDraw.Draw(canvas)
    lines=[]
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(paragraph,width=width) or [""])
    d.multiline_text(xy,"\n".join(lines),font=font(size),fill=fill,spacing=5)
    return xy[1]+len(lines)*(size+5)


def picture(row, size=(900,560), box=None, color="red"):
    with Image.open(row["image_path"]) as f:
        image=f.convert("RGB")
    if box is not None:
        d=ImageDraw.Draw(image)
        coords=[box[0]*image.width,box[1]*image.height,box[2]*image.width,box[3]*image.height]
        if coords[0]<coords[2] and coords[1]<coords[3]:
            d.rectangle(coords,outline=color,width=max(2,image.width//200))
    image.thumbnail(size)
    out=Image.new("RGB",size,"#eeeeee")
    out.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2))
    return out


def records(run_dir):
    result=[]
    for p in sorted((Path(run_dir)/"records").glob("*.jsonl")):
        result.extend(read_jsonl(p))
    return result


def packet(manifest,run_dir,out,kind,geometry_ids=None):
    out.mkdir(parents=True,exist_ok=False)
    rows=read_jsonl(manifest)
    recs=records(run_dir) if run_dir else []
    key=[]
    for row in rows:
        sid=row["sample_id"]
        if geometry_ids is not None and row["image_id"] not in geometry_ids:
            continue
        opaque=hashlib.sha256((kind+sid).encode()).hexdigest()[:12]
        gt=row["ground_truth_bbox"]
        reference=row.get("reference_cot","").splitlines()[-1:]
        if kind in ("eligibility","entity"):
            # Three independently generated entity phrases can be close to the
            # 240-character grammar limit.  Leave enough vertical room so the
            # third trajectory is never clipped from the blind-review card.
            card_height=1400 if kind=="entity" else 1100
            card=Image.new("RGB",(1100,card_height),"white")
            multiline(card,f"Case {opaque}: reference target outlined in red",(25,15),size=21)
            card.paste(picture(row,(1040,570),gt),(30,65))
            y=multiline(card,"Question: "+row["question"],(25,650),width=100)
            y=multiline(card,"Reference: "+" ".join(reference),(25,y+10),width=100)
            if kind=="entity":
                trajectories=sorted([r for r in recs if r.get("type")=="trajectory" and r["sample_id"]==sid],
                                    key=lambda r:r["trajectory_index"])
                for r in trajectories:
                    rid=hashlib.sha256((opaque+str(r["trajectory_index"])).encode()).hexdigest()[:10]
                    label=r.get("entity","") or f"[not generated; {r.get('finish_reason','unknown')}]"
                    y=multiline(card,f"{rid}: {label}",(25,y+15),width=100,size=20)
                    key.append(dict(case_id=opaque,review_id=rid,sample_id=sid,image_id=row["image_id"],
                                    trajectory_index=r["trajectory_index"]))
            else:
                key.append(dict(case_id=opaque,sample_id=sid,image_id=row["image_id"]))
        else:
            card=Image.new("RGB",(1600,1600),"white")
            multiline(card,f"Case {opaque}: reference target (red), candidate predictions (blue)",(20,12),size=23)
            card.paste(picture(row,(620,380),gt),(20,55))
            multiline(card,row["question"],(660,65),width=65,size=21)
            boxes=[r for r in recs if r.get("type")=="bbox" and r.get("stage")=="A" and
                   r.get("mode")=="greedy" and r["sample_id"]==sid]
            boxes.sort(key=lambda r:stable_seed("geometry_blind",sid,r["trajectory_index"],r["arm"]))
            for j,r in enumerate(boxes):
                rid=hashlib.sha256((opaque+str(j)).encode()).hexdigest()[:10]
                x,y=20+(j%4)*395,460+(j//4)*370
                box=[v/1000 for v in r["bbox"]] if r.get("valid") else None
                card.paste(picture(row,(375,300),box,"blue"),(x,y+30))
                multiline(card,rid+ (" [invalid output]" if box is None else ""),(x,y),width=28,size=17)
                key.append(dict(case_id=opaque,review_id=rid,sample_id=sid,image_id=row["image_id"],
                                trajectory_index=r["trajectory_index"],arm=r["arm"],valid=bool(r.get("valid"))))
        card.save(out/(opaque+".jpg"),quality=92)
    # Separate from images; only the coordinator gets this mapping.
    key_path=out.parent/(out.name+"_private_key.jsonl")
    key_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in key))
    instructions=(
        "Review every supplied card independently. Do not read other reviewers or private key files. "
        "Return JSONL {review_id,status,reason}. Entity statuses: correct_unique, wrong_target, ambiguous, "
        "unresolved, uncertain. Correct_unique requires the description to uniquely identify the red-box "
        "target in the image, not merely repeat a broad class or give a correct answer word. "
        "Use uncertain if visual evidence is insufficient. Geometry statuses: same_entity, other_entity, "
        "ambiguous, invalid_output, uncertain; judge blue boxes against the reference target without seeing IoU. "
        "Eligibility: return {case_id,status: eligible|ambiguous|invalid,reason}; verify that the question "
        "and reference identify one object/region and the displayed image is consistent."
    )
    (out/"INSTRUCTIONS.txt").write_text(instructions+"\n")
    return key_path


def reconcile(key_path,review1,review2,output):
    key=read_jsonl(key_path)
    def index(path):
        rows=read_jsonl(path);result={r["review_id"]:r for r in rows}
        if len(result)!=len(rows):
            raise ValueError("duplicate review id")
        return result
    a,b=index(review1),index(review2)
    expected={k["review_id"] for k in key}
    if set(a)!=expected or set(b)!=expected:
        raise ValueError("both independent reviews must cover every expected item")
    allowed={"correct_unique","wrong_target","ambiguous","unresolved","uncertain"}
    result=[]
    for k in key:
        ra,rb=a[k["review_id"]],b[k["review_id"]]
        if ra["status"] not in allowed or rb["status"] not in allowed:
            raise ValueError("invalid entity review status")
        same=ra["status"]==rb["status"]
        result.append(dict(k,status=ra["status"] if same else "uncertain",needs_adjudication=not same,
                           reviewer1=ra,reviewer2=rb))
    with Path(output).open("x") as f:
        for r in result:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest="action",required=True)
    b=s.add_parser("packet")
    b.add_argument("--manifest",type=Path,required=True)
    b.add_argument("--run-dir",type=Path)
    b.add_argument("--output-dir",type=Path,required=True)
    b.add_argument("--kind",choices=("eligibility","entity","geometry"),required=True)
    b.add_argument("--geometry-ids",type=Path)
    c=s.add_parser("reconcile")
    c.add_argument("--key",type=Path,required=True)
    c.add_argument("--review1",type=Path,required=True)
    c.add_argument("--review2",type=Path,required=True)
    c.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.action=="packet":
        ids=set(json.loads(args.geometry_ids.read_text())) if args.geometry_ids else None
        print(packet(args.manifest,args.run_dir,args.output_dir,args.kind,ids))
    else:
        reconcile(args.key,args.review1,args.review2,args.output)


if __name__=="__main__":
    main()
