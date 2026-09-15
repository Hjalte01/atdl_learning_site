import { centers, trajectories, velocity, type Point } from './flow-math';
export function initFlowLab(root:HTMLElement) {
 const select=<T extends Element>(selector:string)=>root.querySelector<T>(selector)!;
 const tabs=Array.from(root.querySelectorAll<HTMLButtonElement>('[data-lab-tab]'));
 const panels=Array.from(root.querySelectorAll<HTMLElement>('[data-lab-panel]'));
 let timer:ReturnType<typeof setInterval>|undefined;
 const play=select<HTMLButtonElement>('[data-flow-play]');
 const stop=()=>{if(timer)clearInterval(timer);timer=undefined;play.textContent='Play generation';};
 const activate=(tab:HTMLButtonElement)=>{
  stop();tabs.forEach(b=>{b.setAttribute('aria-selected',String(b===tab));b.tabIndex=b===tab?0:-1;});
  panels.forEach(p=>p.hidden=p.dataset.labPanel!==tab.dataset.labTab);
  if(tab.dataset.labTab==='sample')draw();
 };
 tabs.forEach((tab,i)=>{
  tab.addEventListener('click',()=>activate(tab));
  tab.addEventListener('keydown',e=>{
   const next=e.key==='ArrowRight'?(i+1)%tabs.length:e.key==='ArrowLeft'?(i+tabs.length-1)%tabs.length:e.key==='Home'?0:e.key==='End'?tabs.length-1:-1;
   if(next>=0){e.preventDefault();activate(tabs[next]);tabs[next].focus();}
  });
 });
 const pair=select<HTMLInputElement>('#pair-time');
 pair.addEventListener('input',()=>{
  const t=Number(pair.value)/100,z=2+8*t;
  select('[data-bridge-marker]').setAttribute('transform',`translate(${80+560*t} 110)`);
  select('[data-bridge-label]').textContent=`z = ${z.toFixed(2)}`;
  select('#pair-readout').textContent=`t = ${t.toFixed(2)}`;
  select('#pair-equation').textContent=`zₜ = ${(1-t).toFixed(2)} × 2 + ${t.toFixed(2)} × 10 = ${z.toFixed(2)}`;
 });
 select<HTMLButtonElement>('[data-reveal-mean]').addEventListener('click',e=>{
  select<HTMLElement>('[data-mean-answer]').hidden=false;
  select('[data-mean-vector]').removeAttribute('visibility');
  const button=e.currentTarget as HTMLButtonElement;button.textContent='Mean velocity revealed';button.disabled=true;
 });
 const canvas=select<HTMLCanvasElement>('[data-field-canvas]'),ctx=canvas.getContext('2d');
 const slider=select<HTMLInputElement>('#sample-step'),stepsSelect=select<HTMLSelectElement>('[data-flow-steps]');
 let steps=24,frame=0,paths=trajectories(steps);
 const project=(point:Point):Point=>[480+point[0]*135,270-point[1]*100];
 function draw(){
  if(!ctx)return;
  const t=1-frame/steps;
  ctx.clearRect(0,0,960,540);ctx.fillStyle='#f4f6ee';ctx.fillRect(0,0,960,540);
  ctx.strokeStyle='#dee4d9';ctx.lineWidth=1;
  for(let x=40;x<960;x+=55){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,540);ctx.stroke();}
  for(let y=20;y<540;y+=55){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(960,y);ctx.stroke();}
  centers.forEach(mu=>{const [x,y]=project(mu);ctx.fillStyle='#16826c13';ctx.strokeStyle='#16826c50';ctx.beginPath();ctx.ellipse(x,y,75,56,0,0,2*Math.PI);ctx.fill();ctx.stroke();});
  // Arrow direction is -v because the sampler decreases t. Length is normalized
  // for legibility, so the illustration does not encode velocity magnitude.
  for(let x=-2.8;x<=2.8;x+=.42)for(let y=-2.2;y<=2.2;y+=.44){
   const v=velocity([x,y],t),length=Math.hypot(v[0],v[1]);if(length<.01)continue;
   const [px,py]=project([x,y]),dx=-v[0]/length*13,dy=v[1]/length*10;
   ctx.strokeStyle='#23806b85';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(px-dx*.4,py-dy*.4);ctx.lineTo(px+dx,py+dy);
   const angle=Math.atan2(dy,dx);ctx.moveTo(px+dx-5*Math.cos(angle-.5),py+dy-5*Math.sin(angle-.5));ctx.lineTo(px+dx,py+dy);ctx.lineTo(px+dx-5*Math.cos(angle+.5),py+dy-5*Math.sin(angle+.5));ctx.stroke();
  }
  paths[0].forEach((_,particle)=>{
   ctx.strokeStyle=particle===0?'#a66a04':'#b8852870';ctx.lineWidth=particle===0?3:1.5;ctx.beginPath();
   for(let i=0;i<=frame;i++){const [x,y]=project(paths[i][particle]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();
   const [x,y]=project(paths[frame][particle]);ctx.beginPath();ctx.arc(x,y,particle===0?7:4,0,2*Math.PI);ctx.fillStyle=particle===0?'#704609':'#ca963b';ctx.fill();
  });
  ctx.fillStyle='#263f35';ctx.font='600 17px system-ui';ctx.fillText('2D toy distribution · generation runs from t = 1 to t = 0',25,30);
  ctx.font='14px system-ui';ctx.fillText('Arrows show direction, not speed. Curves are numerical trajectories.',25,516);
  select('[data-sample-readout]').textContent=`Step ${frame} / ${steps} · t = ${t.toFixed(2)}`;slider.value=String(frame);
  canvas.setAttribute('aria-label',`Toy generation at step ${frame} of ${steps}, time ${t.toFixed(2)}. ${frame===0?'Particles start as Gaussian noise.':frame===steps?'Particles have moved toward the three target regions.':'Particles follow the time-dependent field; their paths may curve.'}`);
 }
 slider.addEventListener('input',()=>{stop();frame=Number(slider.value);draw();});
 select('[data-flow-reset]').addEventListener('click',()=>{stop();frame=0;draw();});
 stepsSelect.addEventListener('change',()=>{stop();steps=Number(stepsSelect.value);frame=0;paths=trajectories(steps);slider.max=String(steps);draw();});
 play.addEventListener('click',()=>{
  if(timer){stop();return;}if(frame===steps)frame=0;
  // Animation is explicitly requested. There is no autoplay. For reduced motion,
  // advance once per click and retain the keyboard-accessible scrubber.
  if(matchMedia('(prefers-reduced-motion: reduce)').matches){frame=Math.min(steps,frame+1);draw();play.textContent='Advance one step';return;}
  play.textContent='Pause';timer=setInterval(()=>{frame=Math.min(steps,frame+1);draw();if(frame===steps)stop();},110);
 });
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
 draw();
}
