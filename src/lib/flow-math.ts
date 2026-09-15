/** Exact conditional velocity for a 2D Gaussian-mixture rectified-flow toy.
 * Convention: z_t=(1-t)x+t*epsilon. Generation integrates t from 1 to 0.
 * Derivation and assumptions: public/docs/flow-lab-math.txt.
 */
export type Point = [number, number];
export const centers: Point[] = [[-1.15,-.75],[1.1,-.6],[0,1.2]];
export const targetVariance = .075;
export function velocity(z: Point, t: number): Point {
  const a=1-t, variance=a*a*targetVariance+t*t;
  const logWeights=centers.map(mu=>-((z[0]-a*mu[0])**2+(z[1]-a*mu[1])**2)/(2*variance));
  const peak=Math.max(...logWeights);
  const weights=logWeights.map(w=>Math.exp(w-peak)), total=weights.reduce((a,b)=>a+b,0);
  const coefficient=(t-a*targetVariance)/variance;
  const out: Point=[0,0];
  centers.forEach((mu,k)=>{for(let j=0;j<2;j++)out[j]+=weights[k]/total*(coefficient*(z[j]-a*mu[j])-mu[j]);});
  return out;
}
export function initialNoise(count=22): Point[] {
  // A reproducible pseudo-random Gaussian sample; identical noise across step counts.
  let state=71931;
  const random=()=>{state=(1664525*state+1013904223)>>>0;return (state+.5)/4294967296;};
  return Array.from({length:count},()=>{const r=Math.sqrt(-2*Math.log(random())),angle=2*Math.PI*random();return [r*Math.cos(angle),r*Math.sin(angle)];});
}
export function trajectories(steps:number): Point[][] {
  const frames:Point[][]=[initialNoise()];
  for(let i=0;i<steps;i++){
    const t=1-i/steps;
    frames.push(frames[i].map(z=>{const v=velocity(z,t);return [z[0]-v[0]/steps,z[1]-v[1]/steps];}));
  }
  return frames;
}
