interface Props {
  title: string;
  detail: string;
  code?: string;
  traceId?: string;
}

export function ErrorBanner({ title, detail, code, traceId }: Props) {
  return (
    <div className="bubble problem" role="alert">
      <strong>{title}</strong>
      <p>{detail}</p>
      {(code || traceId) && (
        <p className="meta">
          {code && <>Code: <code>{code}</code> </>}
          {traceId && <>Trace ID: <code>{traceId}</code></>}
        </p>
      )}
    </div>
  );
}
