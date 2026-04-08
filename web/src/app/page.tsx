"use client";

import { useState } from "react";
import { Database, Brain, Microscope, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import DataPage from "./data/page";
import TrainPage from "./train/page";
import PredictPage from "./predict/page";

const tabs = [
  { id: "data", label: "Data", icon: Database },
  { id: "train", label: "Train", icon: Brain },
  { id: "predict", label: "Predict", icon: Microscope },
] as const;

type TabId = (typeof tabs)[number]["id"];

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("data");
  const { theme, setTheme } = useTheme();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b bg-card sticky top-0 z-50">
        <div className="max-w-[1400px] mx-auto px-6 flex items-center justify-between h-14">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">Ff</span>
            </div>
            <span className="font-semibold text-lg tracking-tight">Foldfit</span>
            <span className="text-xs text-muted-foreground hidden sm:block">
              Antibody Structure Fine-tuning
            </span>
          </div>

          <div className="flex items-center gap-1">
            <nav className="flex bg-muted rounded-lg p-1 gap-0.5">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all cursor-pointer ${
                      isActive
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-background/60"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="hidden sm:inline">{tab.label}</span>
                  </button>
                );
              })}
            </nav>

            <Button
              variant="ghost"
              size="icon"
              className="ml-2"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            </Button>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 max-w-[1400px] mx-auto w-full px-6 py-6">
        {activeTab === "data" && <DataPage />}
        {activeTab === "train" && <TrainPage />}
        {activeTab === "predict" && <PredictPage />}
      </main>
    </div>
  );
}
