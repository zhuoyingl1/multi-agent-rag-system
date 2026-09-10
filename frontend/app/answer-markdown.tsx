"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type AnswerMarkdownProps = {
  content: string;
  citationIds?: string[];
  onCitationClick?: (citationId: string) => void;
};

export default function AnswerMarkdown({ content, citationIds = [], onCitationClick }: AnswerMarkdownProps) {
  const linkedContent = linkKnownCitations(content, citationIds);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a({ href, children }) {
          const citationId = href?.match(/^#source-(S\d+)$/)?.[1];
          if (citationId) {
            return (
              <a
                className="citationLink"
                href={href}
                onClick={(event) => {
                  event.preventDefault();
                  onCitationClick?.(citationId);
                  document.getElementById(`source-${citationId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
                }}
              >
                {children}
              </a>
            );
          }
          return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
        },
      }}
    >
      {linkedContent}
    </ReactMarkdown>
  );
}

function linkKnownCitations(content: string, citationIds: string[]) {
  const knownCitations = new Set(citationIds);
  return content.replace(/\[(S\d+)\]/g, (match, citationId: string) =>
    knownCitations.has(citationId) ? `[${citationId}](#source-${citationId})` : match,
  );
}
