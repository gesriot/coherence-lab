import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { LabApp } from "@/components/lab/lab-app";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("index.html must contain #root");

createRoot(root).render(
  <StrictMode>
    <LabApp />
  </StrictMode>,
);
