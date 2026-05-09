import { FormEvent, useState } from "react";
import { Bot, Send } from "lucide-react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { AssistantMessage } from "@/types/api";

interface AICommandAssistantProps {
  messages: AssistantMessage[];
  onSubmit: (prompt: string) => void;
}

const quickPrompts = ["What should we do?", "Predict escalation", "Generate response strategy"];

export function AICommandAssistant({ messages, onSubmit }: AICommandAssistantProps) {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt) {
      return;
    }
    onSubmit(cleanPrompt);
    setPrompt("");
  };

  return (
    <div className="glass-panel flex h-full flex-col rounded-xl border border-slate-700/60 p-4">
      <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
        <Bot className="h-4 w-4 text-cyan-300" /> AI Command Assistant
      </div>

      <div className="mb-3 flex flex-wrap gap-2">
        {quickPrompts.map((text) => (
          <button
            key={text}
            type="button"
            onClick={() => onSubmit(text)}
            className="rounded-full border border-cyan-400/30 bg-cyan-500/10 px-2 py-1 text-[11px] text-cyan-100 transition hover:bg-cyan-500/20"
          >
            {text}
          </button>
        ))}
      </div>

      <ScrollArea className="mb-3 h-[240px] rounded-lg border border-slate-700/50 bg-slate-950/55 p-3">
        <div className="space-y-2">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "max-w-[90%] rounded-lg px-3 py-2 text-xs",
                message.role === "assistant"
                  ? "border border-cyan-400/25 bg-cyan-500/10 text-cyan-50"
                  : "ml-auto border border-slate-500/50 bg-slate-700/45 text-slate-100"
              )}
            >
              <p>{message.text}</p>
              <p className="mt-1 text-[10px] opacity-70">{message.timestamp}</p>
            </div>
          ))}
        </div>
      </ScrollArea>

      <form onSubmit={handleSubmit} className="mt-auto flex items-center gap-2">
        <Input
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Ask for recommendations..."
          className="h-9"
        />
        <Button size="icon" type="submit">
          <Send className="h-3.5 w-3.5" />
        </Button>
      </form>
    </div>
  );
}
