"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Microscope, Play, Loader2, Sun, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { predictStructure, type PredictResult } from "@/lib/api";

const EXAMPLES = [
  { name: "Trastuzumab VH", sequence: "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS" },
  { name: "Adalimumab VH", sequence: "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS" },
  { name: "Nanobody (VHH)", sequence: "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAIRWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAA" },
];

const REPRESENTATIONS = ["cartoon", "ball+stick", "spacefill", "ribbon", "tube", "surface", "licorice", "backbone"];
const COLOR_SCHEMES = ["chainid", "element", "residueindex", "sstruc", "bfactor", "hydrophobicity", "atomindex", "uniform"];

export default function PredictPage() {
  const [sequence, setSequence] = useState("");
  const [adapterPath, setAdapterPath] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [comparePdbId, setComparePdbId] = useState("");
  const [representation, setRepresentation] = useState("cartoon");
  const [colorScheme, setColorScheme] = useState("chainid");
  const [darkBg, setDarkBg] = useState(false);

  const viewerRef = useRef<HTMLDivElement>(null);
  const compareRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<unknown>(null);
  const compareStageRef = useRef<unknown>(null);

  const loadNgl = useCallback(async (container: HTMLDivElement, pdbId: string, holder: React.MutableRefObject<unknown>) => {
    try {
      const NGL = await import("ngl");
      if (holder.current && typeof (holder.current as { dispose: () => void }).dispose === "function") {
        (holder.current as { dispose: () => void }).dispose();
      }
      const stage = new NGL.Stage(container, { backgroundColor: darkBg ? "#000000" : "#ffffff", quality: "high" });
      holder.current = stage;
      const comp = await stage.loadFile(`rcsb://${pdbId}`, { defaultRepresentation: false }) as { addRepresentation: (t: string, p: Record<string, string>) => void } | undefined;
      if (comp) comp.addRepresentation(representation, { colorScheme });
      stage.autoView();
    } catch {
      toast.error(`Failed to load ${pdbId}`);
    }
  }, [darkBg, representation, colorScheme]);

  const handlePredict = async () => {
    if (!sequence.trim()) { toast.error("Enter a protein sequence"); return; }
    if (!/^[ACDEFGHIKLMNPQRSTVWY]+$/i.test(sequence.replace(/\s/g, ""))) {
      toast.error("Invalid sequence — use standard amino acid letters only"); return;
    }
    setPredicting(true);
    try {
      const res = await predictStructure({
        sequence: sequence.replace(/\s/g, ""),
        adapter_path: adapterPath || undefined,
        device: "cpu",
      });
      setResult(res);
      toast.success(`Structure predicted (${res.sequence_length} residues)`);
      // Load demo PDB in viewer
      if (viewerRef.current) {
        await loadNgl(viewerRef.current, "1IGT", stageRef);
      }
    } catch (e) {
      toast.error(`Prediction failed: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      setPredicting(false);
    }
  };

  const handleCompare = async () => {
    if (!comparePdbId.trim()) { toast.error("Enter a PDB ID to compare"); return; }
    setCompareMode(true);
    setTimeout(() => {
      if (compareRef.current) loadNgl(compareRef.current, comparePdbId.toUpperCase(), compareStageRef);
    }, 100);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      [stageRef, compareStageRef].forEach((ref) => {
        if (ref.current && typeof (ref.current as { dispose: () => void }).dispose === "function") {
          (ref.current as { dispose: () => void }).dispose();
        }
      });
    };
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Predict</h1>
        <p className="text-sm text-muted-foreground mt-1">Predict antibody structures and compare with reference models</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Sequence Input</CardTitle>
              <CardDescription>Paste an antibody sequence or select an example</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Amino Acid Sequence</Label>
                <Textarea
                  placeholder="EVQLVESGGGLVQPGG..."
                  className="font-mono text-xs min-h-[120px] resize-none"
                  value={sequence}
                  onChange={(e) => setSequence(e.target.value.toUpperCase().replace(/[^A-Z]/g, ""))}
                />
                {sequence && <p className="text-xs text-muted-foreground">{sequence.length} residues</p>}
              </div>
              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">Examples</Label>
                <div className="flex flex-wrap gap-1.5">
                  {EXAMPLES.map((ex) => (
                    <button key={ex.name} onClick={() => setSequence(ex.sequence)} className="text-xs px-2.5 py-1 rounded-md border hover:bg-accent transition-colors cursor-pointer">
                      {ex.name}
                    </button>
                  ))}
                </div>
              </div>
              <Separator />
              <div className="space-y-2">
                <Label>LoRA Adapter (optional)</Label>
                <Input placeholder="Path to checkpoint..." value={adapterPath} onChange={(e) => setAdapterPath(e.target.value)} />
                <p className="text-xs text-muted-foreground">Leave empty for base model</p>
              </div>
              <Button onClick={handlePredict} disabled={predicting || !sequence} className="w-full gap-2">
                {predicting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {predicting ? "Predicting..." : "Predict Structure"}
              </Button>
            </CardContent>
          </Card>

          {result && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Result</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex gap-2 flex-wrap">
                  <Badge variant="secondary">{result.sequence_length} residues</Badge>
                  {result.mean_plddt !== null && <Badge variant="outline">pLDDT: {result.mean_plddt?.toFixed(1)}</Badge>}
                </div>
                <Separator />
                <div>
                  <Label className="text-xs text-muted-foreground">Compare with PDB</Label>
                  <div className="flex gap-2 mt-1">
                    <Input placeholder="PDB ID (e.g. 1IGT)" value={comparePdbId} onChange={(e) => setComparePdbId(e.target.value.toUpperCase())} />
                    <Button variant="outline" onClick={handleCompare} className="shrink-0">Load</Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Viewer */}
        <div className={compareMode ? "lg:col-span-1" : "lg:col-span-2"}>
          {result ? (
            <Card className="overflow-hidden">
              <CardHeader className="pb-2 flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-base">Predicted Structure</CardTitle>
                  <CardDescription>{result.sequence_length} residues</CardDescription>
                </div>
                <div className="flex items-center gap-1">
                  <select value={representation} onChange={(e) => setRepresentation(e.target.value)} className="text-xs border rounded px-2 py-1 bg-background">
                    {REPRESENTATIONS.map((r) => <option key={r} value={r}>{r}</option>)}
                  </select>
                  <select value={colorScheme} onChange={(e) => setColorScheme(e.target.value)} className="text-xs border rounded px-2 py-1 bg-background">
                    {COLOR_SCHEMES.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setDarkBg(!darkBg)}>
                    {darkBg ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
                  </Button>
                </div>
              </CardHeader>
              <div ref={viewerRef} className="w-full aspect-[4/3] bg-white dark:bg-black" style={{ minHeight: 400 }} />
            </Card>
          ) : (
            <Card className="flex items-center justify-center aspect-[4/3]" style={{ minHeight: 400 }}>
              <div className="text-center space-y-3">
                <Microscope className="w-12 h-12 text-muted-foreground mx-auto" />
                <h3 className="font-medium text-muted-foreground">No prediction yet</h3>
                <p className="text-sm text-muted-foreground max-w-xs">Enter a sequence and click Predict to visualize the 3D structure</p>
              </div>
            </Card>
          )}
        </div>

        {/* Compare */}
        {compareMode && (
          <div className="lg:col-span-1">
            <Card className="overflow-hidden">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Reference: {comparePdbId}</CardTitle>
                <CardDescription>From PDB</CardDescription>
              </CardHeader>
              <div ref={compareRef} className="w-full aspect-[4/3] bg-white dark:bg-black" style={{ minHeight: 400 }} />
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
