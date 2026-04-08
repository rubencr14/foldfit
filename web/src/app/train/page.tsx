"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Brain, Play, Loader2, ChevronDown, ChevronUp, Zap, RefreshCw, Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import {
  type FinetuneJob,
  type Dataset,
  listFinetuneJobs,
  startFinetune,
  deleteFinetuneJob,
  listDatasets,
} from "@/lib/api";

const statusConfig: Record<string, { color: string; text: string; label: string }> = {
  running: { color: "bg-blue-500", text: "text-blue-700 dark:text-blue-400", label: "Running" },
  completed: { color: "bg-emerald-500", text: "text-emerald-700 dark:text-emerald-400", label: "Completed" },
  failed: { color: "bg-red-500", text: "text-red-700 dark:text-red-400", label: "Failed" },
  queued: { color: "bg-amber-500", text: "text-amber-700 dark:text-amber-400", label: "Queued" },
};

export default function TrainPage() {
  const [jobs, setJobs] = useState<FinetuneJob[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState({
    name: "",
    datasetId: "",
    epochs: 20,
    learningRate: 5e-5,
    loraRank: 8,
    loraAlpha: 16,
    loraDropout: 0,
    scheduler: "cosine",
    warmupSteps: 100,
    accumulationSteps: 4,
    amp: true,
    earlyStoppingPatience: 5,
    maxSeqLen: 256,
    gradClip: 1.0,
  });

  const updateForm = (key: string, value: unknown) => setForm((p) => ({ ...p, [key]: value }));

  const fetchJobs = useCallback(async () => {
    try {
      setLoading(true);
      const [jobsRes, dsRes] = await Promise.all([listFinetuneJobs(), listDatasets()]);
      setJobs(jobsRes.jobs);
      setDatasets(dsRes.datasets);
    } catch (e) {
      toast.error(`Failed to load: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchJobs(); }, [fetchJobs]);

  // Poll running jobs every 5s
  useEffect(() => {
    const hasRunning = jobs.some((j) => j.status === "running" || j.status === "queued");
    if (!hasRunning) return;
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, [jobs, fetchJobs]);

  const handleSubmit = async () => {
    if (!form.name.trim()) { toast.error("Model name is required"); return; }
    if (!form.datasetId) { toast.error("Select a dataset first"); return; }
    setSubmitting(true);
    try {
      const job = await startFinetune({
        name: form.name,
        dataset_id: form.datasetId,
        epochs: form.epochs,
        learning_rate: form.learningRate,
        lora_rank: form.loraRank,
        lora_alpha: form.loraAlpha,
        lora_dropout: form.loraDropout,
        scheduler: form.scheduler,
        warmup_steps: form.warmupSteps,
        accumulation_steps: form.accumulationSteps,
        amp: form.amp,
        early_stopping_patience: form.earlyStoppingPatience,
        max_seq_len: form.maxSeqLen,
        grad_clip: form.gradClip,
      });
      setJobs((prev) => [job, ...prev]);
      setShowForm(false);
      toast.success(`Training job "${job.name}" submitted`);
    } catch (e) {
      toast.error(`Failed to start: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (jobId: string) => {
    try {
      await deleteFinetuneJob(jobId);
      setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
      toast.success("Job deleted");
    } catch (e) {
      toast.error(`Failed to delete: ${e instanceof Error ? e.message : "Unknown error"}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Training</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Fine-tune OpenFold with LoRA on your antibody datasets
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="icon" onClick={fetchJobs} disabled={loading}>
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
          <Button onClick={() => setShowForm(!showForm)} className="gap-2">
            <Zap className="w-4 h-4" /> New Training Job
          </Button>
        </div>
      </div>

      {showForm && (
        <Card className="border-primary/20 bg-primary/[0.02]">
          <CardHeader className="pb-4">
            <CardTitle className="text-base">Configure LoRA Fine-tuning</CardTitle>
            <CardDescription>
              LoRA injects low-rank adapters into Evoformer attention layers (~600K trainable params).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label>Model Name</Label>
                <Input placeholder="e.g. antibody-lora-v1" value={form.name} onChange={(e) => updateForm("name", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Dataset</Label>
                <select
                  className="w-full h-9 rounded-md border bg-background px-3 text-sm"
                  value={form.datasetId}
                  onChange={(e) => updateForm("datasetId", e.target.value)}
                >
                  <option value="">Select dataset...</option>
                  {datasets.map((ds) => (
                    <option key={ds.id} value={ds.id}>{ds.name} ({ds.num_structures})</option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <Label>Epochs</Label>
                <Input type="number" min={1} max={500} value={form.epochs} onChange={(e) => updateForm("epochs", parseInt(e.target.value) || 20)} />
              </div>
              <div className="space-y-2">
                <Label>Learning Rate</Label>
                <Input type="number" step={1e-5} value={form.learningRate} onChange={(e) => updateForm("learningRate", parseFloat(e.target.value) || 5e-5)} />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>LoRA Rank</Label>
                <Input type="number" min={2} max={64} value={form.loraRank} onChange={(e) => updateForm("loraRank", parseInt(e.target.value) || 8)} />
                <p className="text-xs text-muted-foreground">Low-rank dimension (2-64)</p>
              </div>
              <div className="space-y-2">
                <Label>LoRA Alpha</Label>
                <Input type="number" step={1} value={form.loraAlpha} onChange={(e) => updateForm("loraAlpha", parseFloat(e.target.value) || 16)} />
                <p className="text-xs text-muted-foreground">Scaling = {(form.loraAlpha / form.loraRank).toFixed(1)}</p>
              </div>
              <div className="space-y-2">
                <Label>LoRA Dropout</Label>
                <Input type="number" step={0.05} min={0} max={0.5} value={form.loraDropout} onChange={(e) => updateForm("loraDropout", parseFloat(e.target.value) || 0)} />
              </div>
            </div>

            <button onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer">
              {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              Advanced Settings
            </button>

            {showAdvanced && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2 border-t">
                <div className="space-y-2"><Label>Scheduler</Label><Input value={form.scheduler} onChange={(e) => updateForm("scheduler", e.target.value)} /></div>
                <div className="space-y-2"><Label>Warmup Steps</Label><Input type="number" value={form.warmupSteps} onChange={(e) => updateForm("warmupSteps", parseInt(e.target.value) || 0)} /></div>
                <div className="space-y-2"><Label>Grad Accumulation</Label><Input type="number" min={1} value={form.accumulationSteps} onChange={(e) => updateForm("accumulationSteps", parseInt(e.target.value) || 1)} /></div>
                <div className="space-y-2"><Label>Max Seq Length</Label><Input type="number" value={form.maxSeqLen} onChange={(e) => updateForm("maxSeqLen", parseInt(e.target.value) || 256)} /></div>
                <div className="space-y-2"><Label>Grad Clip</Label><Input type="number" step={0.1} value={form.gradClip} onChange={(e) => updateForm("gradClip", parseFloat(e.target.value) || 1.0)} /></div>
                <div className="space-y-2"><Label>Early Stopping</Label><Input type="number" value={form.earlyStoppingPatience} onChange={(e) => updateForm("earlyStoppingPatience", parseInt(e.target.value) || 0)} /></div>
                <div className="flex items-center gap-3 pt-6"><Switch checked={form.amp} onCheckedChange={(v) => updateForm("amp", v)} id="amp" /><Label htmlFor="amp">Mixed Precision</Label></div>
              </div>
            )}

            <div className="flex gap-2 pt-2">
              <Button onClick={handleSubmit} disabled={submitting} className="gap-2">
                {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {submitting ? "Submitting..." : "Start Training"}
              </Button>
              <Button variant="ghost" onClick={() => setShowForm(false)}>Cancel</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {loading && jobs.length === 0 ? (
        <Card className="py-16"><div className="flex flex-col items-center gap-3"><Loader2 className="w-8 h-8 animate-spin text-muted-foreground" /><p className="text-sm text-muted-foreground">Loading jobs...</p></div></Card>
      ) : jobs.length === 0 ? (
        <Card className="py-16"><div className="flex flex-col items-center gap-3 text-center"><Brain className="w-10 h-10 text-muted-foreground" /><h3 className="font-medium">No training jobs</h3><p className="text-sm text-muted-foreground">Start your first LoRA fine-tuning job</p></div></Card>
      ) : (
        <div className="grid gap-3">
          {jobs.map((job) => {
            const sc = statusConfig[job.status] || statusConfig.queued;
            return (
              <Card key={job.job_id} className="hover:border-primary/30 transition-colors">
                <CardContent className="py-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${sc.color} ${job.status === "running" ? "animate-pulse" : ""}`} />
                      <h3 className="font-medium">{job.name}</h3>
                      <Badge variant="outline" className="text-xs">rank={job.lora_rank}</Badge>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <span className={`font-medium ${sc.text}`}>{sc.label}</span>
                      {job.dataset_id && <Badge variant="secondary">{job.dataset_id}</Badge>}
                      <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive hover:text-destructive" onClick={() => handleDelete(job.job_id)}>
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </div>
                  {(job.status === "running" || job.status === "queued") && <Progress value={job.progress} className="h-1.5" />}
                  <div className="flex items-center gap-6 text-sm text-muted-foreground">
                    <span>Epoch {job.epoch}/{job.total_epochs}</span>
                    <span>Train Loss: {job.train_loss.toFixed(4)}</span>
                    {job.val_loss !== null && <span>Val Loss: {job.val_loss.toFixed(4)}</span>}
                    <span className="ml-auto">{job.created_at}</span>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
