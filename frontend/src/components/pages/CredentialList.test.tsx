import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { I18nextProvider } from "react-i18next";

import i18n from "@/i18n";
import { API } from "@/api";
import { CredentialList } from "./CredentialList";

function renderCredentialList(providerId: string) {
  return render(
    <I18nextProvider i18n={i18n}>
      <CredentialList providerId={providerId} />
    </I18nextProvider>,
  );
}

describe("CredentialList", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(API, "listCredentials").mockResolvedValue({ credentials: [] });
  });

  it("shows Base URL input for Jimeng credentials", async () => {
    const user = userEvent.setup();
    renderCredentialList("jimeng");

    await user.click(await screen.findByRole("button", { name: /添加第一个密钥|Add the first key/i }));

    expect(screen.getByLabelText(/Base URL（可选）|Base URL \(optional\)/i)).toBeInTheDocument();
  });

  it("keeps Base URL input hidden for providers without credential base_url support", async () => {
    const user = userEvent.setup();
    renderCredentialList("ark");

    await user.click(await screen.findByRole("button", { name: /添加第一个密钥|Add the first key/i }));

    expect(screen.queryByLabelText(/Base URL（可选）|Base URL \(optional\)/i)).not.toBeInTheDocument();
  });
});
